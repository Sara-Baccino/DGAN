"""
================================================================================
MODULO 5: DGAN.PY
Classe principale con training e generation
================================================================================
"""
import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config.config import DataConfig, DGANConfig, VariableConfig
from generator import HierarchicalGenerator
from discriminator import StaticDiscriminator, TemporalDiscriminator
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class DGAN:
    """DoppelGANger per dati longitudinali."""
    
    def __init__(self, data_config: DataConfig, model_config: DGANConfig):
        self.data_config = data_config
        self.config = model_config
        
        # Device
        if model_config.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Dimensioni
        self.static_dim = (
            len(data_config.static_continuous) +
            sum(len(v.categories) for v in data_config.static_categorical)
        )
        self.temporal_dim = (
            len(data_config.temporal_continuous) +
            sum(len(v.categories) for v in data_config.temporal_categorical)
        )
        
        self._build_model()
        
        # Temperatura Gumbel
        self.current_temperature = model_config.gumbel_temperature_start
        
        # Loss history
        self.loss_history = {
            'generator': [],
            'disc_static': [],
            'disc_temporal': [],
            'gp_static': [],
            'gp_temporal': []
        }
    
    def _build_model(self):
        """Costruisce generator e discriminators."""
        
        self.generator = HierarchicalGenerator(
            data_config=self.data_config,
            z_static_dim=self.config.z_static_dim,
            z_temporal_dim=self.config.z_temporal_dim,
            hidden_dim=self.config.hidden_dim,
            gru_layers=self.config.gru_layers
        ).to(self.device)
        
        self.disc_static = StaticDiscriminator(
            input_dim=self.static_dim,
            num_layers=self.config.discriminator_layers,
            num_units=self.config.discriminator_units
        ).to(self.device)
        
        self.disc_temporal = TemporalDiscriminator(
            static_dim=self.static_dim,
            temporal_dim=self.temporal_dim,
            max_sequence_len=self.data_config.max_sequence_len,
            num_layers=self.config.discriminator_layers,
            num_units=self.config.discriminator_units
        ).to(self.device)
        
        # Optimizers
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_generator,
            betas=(self.config.beta1, 0.999)
        )
        
        self.opt_disc_static = torch.optim.Adam(
            self.disc_static.parameters(),
            lr=self.config.lr_discriminator,
            betas=(self.config.beta1, 0.999)
        )
        
        self.opt_disc_temporal = torch.optim.Adam(
            self.disc_temporal.parameters(),
            lr=self.config.lr_discriminator,
            betas=(self.config.beta1, 0.999)
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
    
    def _combine_features(self, outputs: Dict[str, torch.Tensor]) -> tuple:
        """Combina continuous + categorical."""
        
        # Static
        static_parts = []
        if 'static_continuous' in outputs:
            static_parts.append(outputs['static_continuous'])
        if 'static_categorical' in outputs:
            static_parts.append(outputs['static_categorical'])
        static = torch.cat(static_parts, dim=-1) if static_parts else None
        
        # Temporal
        temporal_parts = []
        if 'temporal_continuous' in outputs:
            temporal_parts.append(outputs['temporal_continuous'])
        if 'temporal_categorical' in outputs:
            temporal_parts.append(outputs['temporal_categorical'])
        temporal = torch.cat(temporal_parts, dim=-1) if temporal_parts else None
        
        mask = outputs.get('temporal_mask')
        
        return static, temporal, mask
    
    def discriminate_static(self, static_features: torch.Tensor) -> torch.Tensor:
        """Valuta static features."""
        return self.disc_static(static_features)
    
    def discriminate_temporal(
        self,
        static_features: torch.Tensor,
        temporal_sequence: torch.Tensor,
        temporal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Valuta temporal condizionato su static."""
        return self.disc_temporal(static_features, temporal_sequence, temporal_mask)
    
    def get_gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: nn.Module,
        additional_inputs: tuple = None
    ) -> torch.Tensor:
        """
        Gradient penalty per WGAN-GP.
        
        Args:
            real: dati reali
            fake: dati generati
            discriminator: discriminator da usare
            additional_inputs: inputs aggiuntivi (per temporal disc)
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        
        # Espandi alpha per matching dimensions
        for _ in range(len(real.shape) - 1):
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.expand_as(real)
        
        # Interpolazione
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        
        # Discriminator output
        if additional_inputs is not None:
            d_interpolated = discriminator(additional_inputs[0], interpolated, additional_inputs[1])
        else:
            d_interpolated = discriminator(interpolated)
        
        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_batch: tuple) -> Dict[str, float]:
        """
        Singolo step di training.

        Training con DUE discriminatori:
        1. Disc_static: valuta solo features statiche
        2. Disc_temporal: valuta sequenze temporali CONDIZIONATE su static

        Supporta:
        - visit_mask corretto
        - visit_times reali
        - initial_states per variabili irreversibili
        """

        (
            static_cont,
            static_cat,
            temporal_cont,
            temporal_cat,
            temporal_mask,
            visit_times,
            initial_states
        ) = real_batch

        # Move to device
        real_batch = tuple(
            x.to(self.device) if x is not None else None
            for x in real_batch
        )

        (
            static_cont,
            static_cat,
            temporal_cont,
            temporal_cat,
            temporal_mask,
            visit_times,
            initial_states
        ) = real_batch

        batch_size = (
            static_cont.size(0)
            if static_cont is not None
            else static_cat.size(0)
        )

        losses = {}

        # ============================================================
        #                   TRAIN DISCRIMINATORS
        # ============================================================
        for _ in range(self.config.discriminator_rounds):

            # -------- Generate fake data --------
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                z_static = torch.randn(
                    batch_size,
                    self.config.z_static_dim,
                    device=self.device
                )
                z_temporal = torch.randn(
                    batch_size,
                    self.data_config.max_sequence_len,
                    self.config.z_temporal_dim,
                    device=self.device
                )

                fake_outputs = self.generator(
                    z_static,
                    z_temporal,
                    self.current_temperature,
                    visit_times=visit_times,
                    initial_states=initial_states
                )

                # Prepare real dict
                real_dict = {
                    'static_continuous': static_cont,
                    'static_categorical': static_cat,
                    'temporal_continuous': temporal_cont,
                    'temporal_categorical': temporal_cat,
                    'temporal_mask': temporal_mask
                }

                real_static, real_temporal, real_mask = self._combine_features(real_dict)
                fake_static, fake_temporal, fake_mask = self._combine_features(fake_outputs)

                fake_static = fake_static.detach()
                fake_temporal = fake_temporal.detach()

            # --------------------------------------------------------
            #               STATIC DISCRIMINATOR
            # --------------------------------------------------------
            self.opt_disc_static.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                d_real_static = self.discriminate_static(real_static)
                d_fake_static = self.discriminate_static(fake_static)

                gp_static = self.get_gradient_penalty(
                    real_static,
                    fake_static,
                    self.disc_static
                )

                loss_d_static = (
                    d_fake_static.mean()
                    - d_real_static.mean()
                    + self.config.gradient_penalty_coef * gp_static
                )

            self.scaler.scale(loss_d_static).backward()

            if self.config.use_dp:
                torch.nn.utils.clip_grad_norm_(
                    self.disc_static.parameters(),
                    self.config.dp_max_grad_norm
                )

            self.scaler.step(self.opt_disc_static)

            # --------------------------------------------------------
            #           TEMPORAL DISCRIMINATOR (condizionato)
            # --------------------------------------------------------
            self.opt_disc_temporal.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                d_real_temporal = self.discriminate_temporal(
                    real_static,
                    real_temporal,
                    real_mask
                )

                d_fake_temporal = self.discriminate_temporal(
                    fake_static,
                    fake_temporal,
                    fake_mask
                )

                gp_temporal = self.get_gradient_penalty(
                    real_temporal,
                    fake_temporal,
                    self.disc_temporal,
                    additional_inputs=(fake_static, fake_mask)
                )

                loss_d_temporal = (
                    d_fake_temporal.mean()
                    - d_real_temporal.mean()
                    + self.config.gradient_penalty_coef * gp_temporal
                )

            self.scaler.scale(loss_d_temporal).backward()

            if self.config.use_dp:
                torch.nn.utils.clip_grad_norm_(
                    self.disc_temporal.parameters(),
                    self.config.dp_max_grad_norm
                )

            self.scaler.step(self.opt_disc_temporal)
            self.scaler.update()

        losses['disc_static'] = loss_d_static.item()
        losses['disc_temporal'] = loss_d_temporal.item()
        losses['gp_static'] = gp_static.item()
        losses['gp_temporal'] = gp_temporal.item()

        # ============================================================
        #                       TRAIN GENERATOR
        # ============================================================
        for _ in range(self.config.generator_rounds):
            self.opt_gen.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                z_static = torch.randn(
                    batch_size,
                    self.config.z_static_dim,
                    device=self.device
                )
                z_temporal = torch.randn(
                    batch_size,
                    self.data_config.max_sequence_len,
                    self.config.z_temporal_dim,
                    device=self.device
                )

                fake_outputs = self.generator(
                    z_static,
                    z_temporal,
                    self.current_temperature,
                    visit_times=visit_times,
                    initial_states=initial_states
                )

                fake_static, fake_temporal, fake_mask = self._combine_features(fake_outputs)

                d_fake_static = self.discriminate_static(fake_static)
                d_fake_temporal = self.discriminate_temporal(
                    fake_static,
                    fake_temporal,
                    fake_mask
                )

                # Generator loss: fool BOTH discriminators
                loss_g = -(d_fake_static.mean() + d_fake_temporal.mean())

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.opt_gen)
            self.scaler.update()

        losses['generator'] = loss_g.item()

        return losses

        
    def update_losses(self, losses: Dict[str, float]):
        """Aggiorna history."""
        for key, value in losses.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
    
    def fit(
        self,
        train_data: tuple,
        validation_data: Optional[tuple] = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Training completo.
        
        Args:
            train_data: tuple di tensori (static_cont, static_cat, temporal_cont, temporal_cat, mask)
            validation_data: opzionale per validazione
            progress_callback: callback(epoch, batch, total_batches, losses)
        """
        
        # Crea dataset
        dataset = TensorDataset(*[x for x in train_data if x is not None])
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Set training mode
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            epoch_losses = {key: [] for key in self.loss_history.keys()}
            
            for batch_idx, batch in enumerate(dataloader):
                losses = self.train_step(batch)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
                
                if progress_callback is not None:
                    progress_callback(epoch, batch_idx, len(dataloader), losses)
            
            # Media epoch
            for key in epoch_losses:
                mean_loss = np.mean(epoch_losses[key])
                self.loss_history[key].append(mean_loss)
            
            # Decay temperatura Gumbel
            self.current_temperature = max(
                self.config.gumbel_temperature_end,
                self.current_temperature * self.config.gumbel_temperature_decay
            )
            
            # Validazione
            if validation_data is not None and epoch % 10 == 0:
                val_metrics = self.validate(validation_data)
                logger.info(f"Epoch {epoch} - Validation: {val_metrics}")
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"G: {self.loss_history['generator'][-1]:.4f}, "
                    f"D_static: {self.loss_history['disc_static'][-1]:.4f}, "
                    f"D_temporal: {self.loss_history['disc_temporal'][-1]:.4f}, "
                    f"Temp: {self.current_temperature:.3f}"
                )
    
    def validate(self, val_data: tuple) -> Dict[str, float]:
        """Validazione."""
        
        self.generator.eval()
        self.disc_static.eval()
        self.disc_temporal.eval()
        
        dataset = TensorDataset(*[x for x in val_data if x is not None])
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        metrics = {
            'd_static_real': [],
            'd_static_fake': [],
            'd_temporal_real': [],
            'd_temporal_fake': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(x.to(self.device) if x is not None else None for x in batch)
                batch_size = batch[0].size(0) if batch[0] is not None else batch[1].size(0)
                
                # Real
                real_dict = {
                    'static_continuous': batch[0],
                    'static_categorical': batch[1],
                    'temporal_continuous': batch[2],
                    'temporal_categorical': batch[3],
                    'temporal_mask': batch[4]
                }
                real_static, real_temporal, real_mask = self._combine_features(real_dict)
                
                # Fake
                z_static = torch.randn(batch_size, self.config.z_static_dim, device=self.device)
                z_temporal = torch.randn(
                    batch_size,
                    self.data_config.max_sequence_len,
                    self.config.z_temporal_dim,
                    device=self.device
                )
                fake_outputs = self.generator(z_static, z_temporal, self.current_temperature)
                fake_static, fake_temporal, fake_mask = self._combine_features(fake_outputs)
                
                # Scores
                metrics['d_static_real'].append(self.discriminate_static(real_static).mean().item())
                metrics['d_static_fake'].append(self.discriminate_static(fake_static).mean().item())
                metrics['d_temporal_real'].append(
                    self.discriminate_temporal(real_static, real_temporal, real_mask).mean().item()
                )
                metrics['d_temporal_fake'].append(
                    self.discriminate_temporal(fake_static, fake_temporal, fake_mask).mean().item()
                )
        
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def generate(
        self,
        n_samples: int,
        return_torch: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Genera campioni sintetici.
        
        Args:
            n_samples: numero di campioni
            return_torch: se True ritorna tensori torch, altrimenti numpy
        
        Returns:
            dict con outputs del generator
        """
        self.generator.eval()
        
        all_outputs = []
        n_generated = 0
        
        with torch.no_grad():
            while n_generated < n_samples:
                batch_size = min(self.config.batch_size, n_samples - n_generated)
                
                z_static = torch.randn(batch_size, self.config.z_static_dim, device=self.device)
                z_temporal = torch.randn(
                    batch_size,
                    self.data_config.max_sequence_len,
                    self.config.z_temporal_dim,
                    device=self.device
                )
                
                outputs = self.generator(z_static, z_temporal, temperature=0.5)
                all_outputs.append(outputs)
                n_generated += batch_size
        
        # Concatena batches
        final_outputs = {}
        for key in all_outputs[0].keys():
            tensors = [batch[key] for batch in all_outputs if batch[key] is not None]
            if tensors:
                concatenated = torch.cat(tensors, dim=0)[:n_samples]
                if return_torch:
                    final_outputs[key] = concatenated
                else:
                    final_outputs[key] = concatenated.cpu().numpy()
        
        self.generator.train()
        
        return final_outputs
    
    def save(self, filepath: str):
        """Salva modello."""
        state = {
            'data_config': {
                'variables': [vars(v) for v in self.data_config.variables],
                'max_sequence_len': self.data_config.max_sequence_len
            },
            'model_config': vars(self.config),
            'generator_state': self.generator.state_dict(),
            'disc_static_state': self.disc_static.state_dict(),
            'disc_temporal_state': self.disc_temporal.state_dict(),
            'opt_gen_state': self.opt_gen.state_dict(),
            'opt_disc_static_state': self.opt_disc_static.state_dict(),
            'opt_disc_temporal_state': self.opt_disc_temporal.state_dict(),
            'loss_history': self.loss_history,
            'current_temperature': self.current_temperature
        }
        torch.save(state, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'DGAN':
        """Carica modello."""
        state = torch.load(filepath, map_location='cpu')
        
        # Ricostruisci configs
        variables = [VariableConfig(**v) for v in state['data_config']['variables']]
        data_config = DataConfig(
            variables=variables,
            max_sequence_len=state['data_config']['max_sequence_len']
        )
        
        model_config = DGANConfig(**state['model_config'])
        if device is not None:
            model_config.cuda = (device == 'cuda')
        
        # Crea modello
        dgan = cls(data_config, model_config)
        
        # Carica state dicts
        dgan.generator.load_state_dict(state['generator_state'])
        dgan.disc_static.load_state_dict(state['disc_static_state'])
        dgan.disc_temporal.load_state_dict(state['disc_temporal_state'])
        dgan.opt_gen.load_state_dict(state['opt_gen_state'])
        dgan.opt_disc_static.load_state_dict(state['opt_disc_static_state'])
        dgan.opt_disc_temporal.load_state_dict(state['opt_disc_temporal_state'])
        dgan.loss_history = state['loss_history']
        dgan.current_temperature = state['current_temperature']
        
        logger.info(f"Model loaded from {filepath}")
        return dgan

