"""
model/generator.py  
================================================================================
t_FUP vincolato: t_FUP >= last_visit_time
    Architettura a 3 step:
        a) followup_head(s_h) → t_FUP_norm ∈ [0,1]   (durata totale follow-up)
        b) loop AR → Δt_raw[t] per t=0..T-1
        c) cumsum(Δt_raw[:n_visits]) = T_obs (durata osservazioni)
           se T_obs > t_FUP: scala tutti i Δt per t_FUP/T_obs
           → garantisce T_obs <= t_FUP per costruzione, senza clipping brusco

Categoriche temporali nell'autoregression
      L'input GRU include anche le OHE delle categoriche temporali del passo
      precedente, oltre ai continui e al delta. Questo permette al modello di
      apprendere transizioni categoriche realistiche (es. DEATH irreversibile).

Self-Attention dopo la GRU
      Un singolo layer di Multi-Head Self-Attention (4 teste, hidden_dim dim)
      applicato all'output h_seq del GRU prima degli head di output.
      Permette a ogni step di "guardare" tutti gli altri step validi,
      catturando dipendenze a lungo raggio (es. risposta UDCA dopo mese 12).

Proiezione di s_h concatenata all'input GRU a ogni step
      Input GRU: [z_t, x_prev_cont, cat_prev_ohe, delta_prev, s_h_proj]
      s_h_proj = Linear(hidden_dim → static_proj_dim) applicato una volta,
      poi broadcastato su tutti gli step.
      Questo inietta il contesto statico del paziente a ogni passo temporale,
      invece di affidarsi solo all'hidden state iniziale h0.


- Gumbel-Softmax per categoriche statiche e temporali
- cummax per variabili irreversibili
- noise AR configurabile
- valid_flag da n_visits_head
================================================================================
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DGANGenerator(nn.Module):
    """Generatore DGAN.
    """

    def __init__(
        self,
        data_config,
        preprocessor,
        z_static_dim:    int   = 64,
        z_temporal_dim:  int   = 48,
        hidden_dim:      int   = 64,
        n_layers:        int   = 2,
        dropout:         float = 0.1,
        noise_ar_rho:    float = 0.0,
        min_visits:      int   = 2,
        static_proj_dim: int   = 16,   # proiezione s_h → input GRU ogni step
        attn_heads:      int   = 4,    # teste Self-Attention (0 = skip)
    ):
        super().__init__()
        self.data_config     = data_config
        self.preprocessor    = preprocessor
        self.max_len         = data_config.max_len
        self.hidden_dim      = hidden_dim
        self.z_static_dim    = z_static_dim
        self.z_temporal_dim  = z_temporal_dim
        self.n_layers        = n_layers
        self.noise_ar_rho    = float(noise_ar_rho)
        self.min_visits      = max(1, int(min_visits))
        self.static_cond_dim = static_proj_dim
        self.attn_heads      = attn_heads

        n_cont = data_config.n_temp_cont

        # ── [1] Static branch ──────────────────────────────────────────
        # s_h è la rappresentazione densa del paziente.
        # Usata per: h0, static outputs, followup, n_visits, static_cond a ogni step.
        self.fc_static = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.to_h0 = nn.Linear(hidden_dim, n_layers * hidden_dim)

        # ── t_FUP dal ramo statico ─────────────────────────────────────
        # Produce t_FUP_norm ∈ [0,1]. Questa è la durata TOTALE del follow-up.
        # I visit_times generati saranno scalati per non superare t_FUP.
        self.followup_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),   # input: s_h, non z_static
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.followup_head[-2].bias, -0.62)

        # n_visits dal ramo statico
        self.n_visits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),   # input: s_h
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.n_visits_head[-1].bias, 4.0)

        # ── Static outputs da s_h (non da h_seq) ──────────────────────
        # [1] Più lineare: s_h è già la repr. del paziente, non dipende
        # dall'evoluzione temporale della GRU.
        if data_config.n_static_cont > 0:
            self.static_cont_head = nn.Linear(hidden_dim, data_config.n_static_cont)
            nn.init.zeros_(self.static_cont_head.weight)
            nn.init.zeros_(self.static_cont_head.bias)
        else:
            self.static_cont_head = None

        self.static_cat_heads = nn.ModuleDict()
        for v in data_config.static_cat:
            if v.name in preprocessor.embedding_configs:
                self.static_cat_heads[v.name] = nn.Linear(
                    hidden_dim, preprocessor.embedding_configs[v.name])
            else:
                if v.n_categories is None or v.n_categories < 2:
                    raise ValueError(
                        f"Categorica statica '{v.name}' ha mapping invalido "
                        f"(n_categories={v.n_categories})."
                    )
                self.static_cat_heads[v.name] = nn.Linear(hidden_dim, v.n_categories)

        # ── [6] Proiezione statica per conditioning a ogni step ────────
        # s_h_proj viene concatenata all'input GRU a ogni t.
        # Inietta il contesto paziente in ogni passo senza affidarsi solo a h0.
        self.static_cond_proj = nn.Linear(hidden_dim, static_proj_dim)

        # ── Calcola dim OHE temporale per l'input GRU ─────────────────
        # [3] Categorie temporali nell'autoregression
        self._temp_cat_sizes: Dict[str, int] = {}  # {name: n_categories}
        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue
            if var.name not in [v.name for v in data_config.temporal_cat]:
                continue
            if var.mapping and len(var.mapping) >= 2:
                if var.irreversible:
                    self._temp_cat_sizes[var.name] = 2   # OHE binaria (0/1)
                else:
                    self._temp_cat_sizes[var.name] = len(var.mapping)

        total_cat_prev_dim = sum(self._temp_cat_sizes.values())

        # ── [4] Temporal GRU ───────────────────────────────────────────
        # input: [z_t, x_prev_cont, cat_prev_ohe, delta_prev, s_h_proj]
        self._n_cont_input = max(n_cont, 0)
        gru_input_size = (
            z_temporal_dim
            + self._n_cont_input       # continui passo precedente
            + total_cat_prev_dim       # OHE categorici passo precedente [3]
            + 1                        # delta_prev
            + static_proj_dim          # proiezione statica [6]
        )

        self.gru = nn.GRU(
            input_size  = gru_input_size,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        # ── [5] Self-Attention post-GRU ───────────────────────────────
        # Applicato a h_seq prima degli head di output.
        # embed_dim deve essere divisibile per attn_heads.
        if attn_heads > 0:
            # Assicura divisibilità
            actual_heads = attn_heads
            while hidden_dim % actual_heads != 0 and actual_heads > 1:
                actual_heads -= 1
            self.self_attn = nn.MultiheadAttention(
                embed_dim   = hidden_dim,
                num_heads   = actual_heads,
                dropout     = dropout,
                batch_first = True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
        else:
            self.self_attn = None
            self.attn_norm = None

        # ── Output temporali ───────────────────────────────────────────
        out_cont_dim = max(n_cont, 1)
        self.temporal_cont_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_cont_dim),
        )
        nn.init.zeros_(self.temporal_cont_head[-1].weight)
        nn.init.zeros_(self.temporal_cont_head[-1].bias)

        # Intervallo inter-visita Δt ≥ 0 (softplus)
        self.interval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Bias iniziale: softplus(0.5) ≈ 0.97 → intervallo medio moderato
        nn.init.constant_(self.interval_head[-1].bias, 0.5)

        self.temporal_cat_heads = nn.ModuleDict()
        self.temporal_cat_irrev: Dict[str, bool] = {}
        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue
            if var.name not in [v.name for v in data_config.temporal_cat]:
                continue
            if var.mapping is None or len(var.mapping) < 2:
                raise ValueError(
                    f"Categorica temporale '{var.name}' ha mapping invalido."
                )
            if var.irreversible:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, 1)
                self.temporal_cat_irrev[var.name] = True
            else:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, len(var.mapping))
                self.temporal_cat_irrev[var.name] = False

    # ------------------------------------------------------------------

    @staticmethod
    def _cummax_irr(hazard: torch.Tensor) -> torch.Tensor:
        states, _ = torch.cummax(hazard, dim=1)
        return states

    def sample_noise(self, batch_size: int, device: torch.device) -> tuple:
        z_s = torch.randn(batch_size, self.z_static_dim, device=device)
        if self.noise_ar_rho <= 0.0:
            z_t = torch.randn(batch_size, self.max_len, self.z_temporal_dim, device=device)
        else:
            rho       = self.noise_ar_rho
            std_innov = (1.0 - rho ** 2) ** 0.5
            z_t       = torch.zeros(batch_size, self.max_len, self.z_temporal_dim, device=device)
            z_t[:, 0, :] = torch.randn(batch_size, self.z_temporal_dim, device=device)
            for t in range(1, self.max_len):
                eps           = torch.randn(batch_size, self.z_temporal_dim, device=device)
                z_t[:, t, :] = rho * z_t[:, t - 1, :] + std_innov * eps
        return z_s, z_t

    # ------------------------------------------------------------------

    def forward(
        self,
        z_static:    torch.Tensor,              # [B, z_s]
        z_temporal:  torch.Tensor,              # [B, T, z_t]
        temperature: float                    = 1.0,
        real_irr:    Optional[torch.Tensor]   = None,
    ) -> Dict:
        B, T, _ = z_temporal.shape
        device   = z_temporal.device
        n_cont   = self.data_config.n_temp_cont

        # ── Step 1: Ramo statico ───────────────────────────────────────
        s_h = self.fc_static(z_static)                          # [B, H]
        h0  = self.to_h0(s_h)                                   # [B, n_layers*H]
        h0  = h0.view(B, self.n_layers, self.hidden_dim) \
                 .permute(1, 0, 2).contiguous()                 # [n_layers, B, H]

        # [1] Static outputs da s_h (non dalla GRU)
        static_cont = None
        if self.static_cont_head is not None:
            static_cont = self.static_cont_head(s_h)           # [B, n_static_cont]

        static_cat       = {}
        static_cat_soft  = {}
        static_cat_embed = {}
        embed_cfg        = self.preprocessor.embedding_configs

        for name, head in self.static_cat_heads.items():
            out = head(s_h)                                     # [B, n_cat]
            if name in embed_cfg:
                static_cat_embed[name] = out
            else:
                static_cat[name]      = F.gumbel_softmax(out, tau=temperature, hard=True,  dim=-1)
                static_cat_soft[name] = F.gumbel_softmax(out, tau=temperature, hard=False, dim=-1)

        # Scalari globali dal ramo statico
        followup_norm = self.followup_head(s_h).squeeze(-1)     # [B] ∈ [0,1]  (t_FUP)
        n_v_raw       = F.softplus(self.n_visits_head(s_h).squeeze(-1)) + 1.0
        n_visits      = n_v_raw.clamp(float(self.min_visits), float(T))  # [B]

        # valid_flag: True per i primi n_visits step
        t_pos     = torch.arange(T, dtype=torch.float32, device=device)
        n_v_round = n_visits.round().long().clamp(self.min_visits, T)     # [B]
        valid_flag = t_pos.unsqueeze(0) < n_v_round.float().unsqueeze(1)  # [B,T]

        # [6] Proiezione statica da concatenare a ogni step GRU
        s_cond = self.static_cond_proj(s_h)                    # [B, static_cond_dim]

        # ── Step 2: Loop AR pre-allocato ───────────────────────────────
        # [4] Pre-alloca tutti i buffer invece di append a liste
        h_seq           = torch.zeros(B, T, self.hidden_dim, device=device)
        deltas_buf      = torch.zeros(B, T, device=device)
        cont_buf        = torch.zeros(B, T, max(n_cont, 1), device=device)

        # Pre-alloca buffer per categoriche temporali
        cat_logit_bufs: Dict[str, torch.Tensor] = {}
        for name, n_cat in self._temp_cat_sizes.items():
            cat_logit_bufs[name] = torch.zeros(B, T, n_cat, device=device)

        # Stato iniziale del loop
        x_prev_cont  = torch.zeros(B, self._n_cont_input, device=device)
        cat_prev_ohe = torch.zeros(B, sum(self._temp_cat_sizes.values()), device=device)
        delta_prev   = torch.zeros(B, 1, device=device)
        hidden       = h0

        for t in range(T):
            # [6] Input GRU: [z_t, x_prev_cont, cat_prev_ohe, delta_prev, s_cond]
            gru_in = torch.cat(
                [z_temporal[:, t, :], x_prev_cont, cat_prev_ohe, delta_prev, s_cond],
                dim=-1,
            ).unsqueeze(1)                                      # [B, 1, input_size]

            h_t_3d, hidden = self.gru(gru_in, hidden)          # [B, 1, H]
            h_t = h_t_3d.squeeze(1)                            # [B, H]
            h_seq[:, t, :] = h_t

            # Continui: output + aggiorna x_prev_cont
            x_cont_t = self.temporal_cont_head(h_t)[:, :n_cont] if n_cont > 0 \
                       else h_t.new_zeros(B, 0)
            cont_buf[:, t, :n_cont] = x_cont_t if n_cont > 0 else cont_buf[:, t, :n_cont]
            if n_cont > 0:
                x_prev_cont = x_cont_t.detach()                # stop-gradient

            # Delta Δt
            delta_t = F.softplus(self.interval_head(h_t))      # [B, 1]
            deltas_buf[:, t] = delta_t.squeeze(-1)
            delta_prev = delta_t.detach()

            # [3] Categoriche temporali: logit + aggiorna cat_prev_ohe
            cat_parts = []
            offset = 0
            for name, n_cat in self._temp_cat_sizes.items():
                logit_t = cat_logit_bufs[name][:, t, :]        # view dello slice
                logit_t_new = self.temporal_cat_heads[name](h_t)   # [B, n_cat or 1]
                if self.temporal_cat_irrev.get(name, False):
                    # irrev: usa sigmoid → probabilità di evento
                    p = torch.sigmoid(logit_t_new).squeeze(-1)  # [B]
                    ohe_t = torch.stack([1 - p, p], dim=-1)     # [B, 2]
                    cat_logit_bufs[name][:, t, :] = ohe_t.detach()
                    cat_parts.append(ohe_t.detach())
                else:
                    cat_logit_bufs[name][:, t, :] = logit_t_new.detach()
                    # Soft OHE per il conditioning del passo successivo
                    soft_ohe = F.gumbel_softmax(logit_t_new.detach(), tau=temperature,
                                                hard=False, dim=-1)
                    cat_parts.append(soft_ohe)
                offset += n_cat

            if cat_parts:
                cat_prev_ohe = torch.cat(cat_parts, dim=-1)    # [B, sum(n_cat)]

        # ── Step 3: Self-Attention post-GRU ───────────────────────────
        # [5] Applica multi-head self-attention con key_padding_mask per il padding
        if self.self_attn is not None:
            # key_padding_mask: True = posizioni DA IGNORARE (padding)
            # valid_flag: True = valida → padding = ~valid_flag
            attn_mask_key = ~valid_flag                         # [B, T] bool
            attn_out, _ = self.self_attn(
                h_seq, h_seq, h_seq,
                key_padding_mask = attn_mask_key,
            )
            # Residual + LayerNorm
            h_seq = self.attn_norm(h_seq + attn_out)

        # ── Step 4: Output temporali finali ───────────────────────────
        # Re-calcola i continui dall'h_seq post-attention per output finale
        # (durante il loop usavamo h_t pre-attention; qui usiamo h_seq post-attention)
        temporal_cont_final = self.temporal_cont_head(h_seq)   # [B, T, out_dim]
        temporal_cont = temporal_cont_final[:, :, :n_cont] if n_cont > 0 \
                        else temporal_cont_final[:, :, :0]

        # Categoriche temporali finali dall'h_seq post-attention
        temporal_cat = {}
        irrev_names  = [n for n, irr in self.temporal_cat_irrev.items() if irr]

        for name, head in self.temporal_cat_heads.items():
            if self.temporal_cat_irrev[name]:
                hazard = torch.sigmoid(head(h_seq).squeeze(-1))   # [B, T]
                if real_irr is not None and len(irrev_names) > 0:
                    try:
                        irr_idx    = irrev_names.index(name)
                        irr_states = real_irr[:, :, irr_idx].float()
                    except (ValueError, IndexError):
                        irr_states = self._cummax_irr(hazard)
                else:
                    irr_states = self._cummax_irr(hazard)
                temporal_cat[name] = torch.stack([1.0 - irr_states, irr_states], dim=-1)
            else:
                temporal_cat[name] = F.gumbel_softmax(
                    head(h_seq), tau=temperature, hard=True, dim=-1)

        # ── Step 5: Costruzione visit_times con vincolo t_FUP ─────────
        
        # [2] Architettura in 3 step:
        #   a) t_FUP già generato: followup_norm [0,1]
        #   b) cumsum(deltas) per le visite valide → T_obs (durata osservazioni)
        #   c) se T_obs > t_FUP: scala i delta per t_FUP/T_obs
        #      → garantisce visit_last <= t_FUP per costruzione

        # Cumsum dei delta
        cumsum_d = torch.cumsum(deltas_buf, dim=1)              # [B, T]

        # T_obs = cumsum ultima visita valida
        last_idx = (n_v_round - 1).clamp(0, T-1).unsqueeze(1)
        T_obs = cumsum_d.gather(1, last_idx).squeeze(1)  # [B]
        T_obs = T_obs.clamp(min=1e-6)

        # Ratio scala ≤ 1
        ratio = torch.minimum(followup_norm / T_obs, torch.ones_like(followup_norm))
        delta_final = deltas_buf * ratio.unsqueeze(1)  # [B,T]

        # visit_times normalizzati [0,1]
        visit_times = torch.cumsum(delta_final, dim=1) / followup_norm.clamp(min=1e-6).unsqueeze(1)
        visit_times = visit_times.clamp(0.0, 1.0)
        
        '''
        # ── Step 5: Costruzione visit_times con logica Capping (NO compressione) ──
        
        # 1. Calcoliamo i tempi di visita "naturali" (cumsum dei delta generati dalla GRU)
        # Sottraiamo il primo delta se vogliamo che la visita 0 sia al tempo 0.
        natural_visit_times = torch.cumsum(deltas_buf, dim=1) - deltas_buf[:, 0:1] 

        # 2. Otteniamo t_FUP_norm (il limite massimo generato dal ramo statico)
        t_fup = followup_norm.unsqueeze(1) # [B, 1]

        # 3. CAPPING: Se la visita naturale supera t_FUP, viene bloccata a t_FUP.
        # Questo garantisce visit_times <= t_FUP senza modificare le visite precedenti.
        visit_times = torch.min(natural_visit_times, t_fup).clamp(0.0, 1.0)

        # 4. Ricalcolo dei delta coerenti con i tempi cappati (per l'output)
        delta_final = torch.cat([
            visit_times[:, 0:1], 
            visit_times[:, 1:] - visit_times[:, :-1]
        ], dim=1)
        '''

        return {
            "static_cont":      static_cont,
            "static_cat":       static_cat      if static_cat      else None,
            "static_cat_soft":  static_cat_soft if static_cat_soft else {},
            "static_cat_embed": static_cat_embed if static_cat_embed else None,
            "temporal_cont":    temporal_cont,
            "temporal_cat":     temporal_cat,
            "valid_flag":       valid_flag,          # [B,T] bool
            "visit_times":      visit_times,         # [B,T] ∈ [0,1], scala t_FUP
            "deltas":           delta_final,            #delta_scaled,        # [B,T] intervalli scalati
            "followup_norm":    followup_norm,       # [B] ∈ [0,1] — t_FUP
            "n_visits":         n_visits,
            "n_visits_pred":    n_v_raw,
        }