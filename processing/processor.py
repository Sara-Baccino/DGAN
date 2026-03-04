"""
processing/preprocessor.py
================================================================================
Modifiche rispetto alla versione precedente:

  [v4] Normalizzazione temporale a due livelli (per-paziente):

    PROBLEMA CON SCHEMA GLOBALE:
      global_time_max ≈ 240 mesi (paziente con 20 anni di follow-up)
      Paziente con 2 visite in 6 mesi → t_norm = [0, 0.025]
      Il GRU riceve delta_t ≈ 0 per quasi tutti i pazienti corti.
      La struttura temporale viene persa per il 90%+ dei pazienti.

    NUOVO SCHEMA A DUE LIVELLI:
      t_norm[i,t]      = t_raw[i,t] / t_max[i]           ∈ [0,1] per-paziente
      followup_norm[i] = t_max[i]   / global_time_max     ∈ [0,1]

      Il GRU vede sempre [0,1] indipendentemente dalla durata assoluta.
      followup_norm cattura la durata assoluta normalizzata:
        - paziente con 6 mesi su 240: followup_norm = 0.025
        - paziente con 20 anni:       followup_norm = 1.0
      In inverse_transform: t_abs = t_norm × followup_norm × global_time_max

  [v4] followup_norm aggiunto ai tensori di output (dtype float32, shape [N]).
       Viene passato al discriminatore come feature temporale broadcastata
       e usato dal followup_head del generatore come target implicito.

  I bug fix delle versioni precedenti (missing sentinel, validate_mappings,
  BUG 5 ordine static_cat, BUG 6 expm1 static, BUG 8 valid_steps) sono
  tutti mantenuti.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from config.config_loader import DataConfig

MAP_MISSING = "__MISSING__"


class Preprocessor:

    def __init__(
        self,
        data_cfg:          DataConfig,
        embedding_configs: Optional[Dict[str, int]] = None,
        log_vars:          Optional[List[str]] = None,
    ):
        self.vars = (
            data_cfg.static_cont
            + data_cfg.static_cat
            + data_cfg.temporal_cont
            + data_cfg.temporal_cat
        )
        self.log_vars          = log_vars or []
        self.max_len           = data_cfg.max_len
        self.id_col            = data_cfg.patient_id_col
        self.time_col          = data_cfg.time_col
        self.embedding_configs = embedding_configs or {}
        self.embeddings        = nn.ModuleDict()
        self.scalers_cont      = {}
        self.inverse_maps      = {}
        self.global_time_max   = None  # impostato in normalize_time_per_patient

        self._validate_mappings()

    # ------------------------------------------------------------------
    def _validate_mappings(self):
        """Encoding 0 riservato per missing. Lancia ValueError se violato."""
        for v in self.vars:
            if v.kind != "categorical":
                continue
            if v.mapping and 0 in v.mapping.values():
                raise ValueError(
                    f"Variable '{v.name}': encoding 0 è riservato per i missing. "
                    f"Tutti gli encoding devono partire da 1. Mapping: {v.mapping}"
                )

    # ==================================================================
    # FIT + TRANSFORM
    # ==================================================================

    def fit_transform(self, df: pd.DataFrame) -> Dict:
        df               = df.reset_index(drop=True)
        df               = self.force_types(df)
        df, cat_masks    = self.encode_categoricals(df)
        df, cont_masks   = self.process_continuous(df)
        padded           = self.long_to_padded(df, cat_masks, cont_masks)
        padded           = self.fit_scalers(padded)
        padded           = self.normalize_time_per_patient(padded)   # [v4]
        static_out       = self.build_static_tensors(padded["df_static"])
        padded.update(static_out)
        return self.to_tensors(padded)

    # ==================================================================
    # FORCE TYPES
    # ==================================================================

    def force_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte i tipi. Per categoriche: NaN/None/'' → MAP_MISSING
        prima di astype(str), altrimenti pd.NA diventa "nan" e sfugge
        al controllo missing in encode_categoricals.
        """
        df        = df.copy()
        _STR_NANS = {"nan", "none", "<na>", "nat", ""}

        for v in self.vars:
            if v.name not in df.columns:
                continue
            if v.kind == "continuous":
                df[v.name] = pd.to_numeric(df[v.name], errors="coerce")
            else:
                col = df[v.name].astype(str)
                col = col.where(~col.str.lower().isin(_STR_NANS), other=MAP_MISSING)
                col = col.fillna(MAP_MISSING)
                df[v.name] = col
        return df

    # ==================================================================
    # ENCODE CATEGORICALS
    # ==================================================================

    def encode_categoricals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df    = df.copy()
        masks = {}
        for v in self.vars:
            if v.kind != "categorical" or v.name not in df.columns:
                continue
            self.inverse_maps[v.name] = {val: key for key, val in v.mapping.items()}
            col           = df[v.name]
            masks[v.name] = (col != MAP_MISSING).astype(float).values
            df[v.name]    = col.map(lambda x, m=v.mapping: m.get(x, 0)).astype(int)
        return df, masks

    # ==================================================================
    # PROCESS CONTINUOUS
    # ==================================================================

    def process_continuous(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df    = df.copy()
        masks = {}
        for v in self.vars:
            if v.kind != "continuous" or v.name not in df.columns:
                continue
            masks[v.name] = (~df[v.name].isna()).astype(float).values
        return df, masks

    # ==================================================================
    # LONG → PADDED
    # ==================================================================

    def long_to_padded(
        self,
        df:         pd.DataFrame,
        cat_masks:  Dict[str, np.ndarray],
        cont_masks: Dict[str, np.ndarray],
    ) -> Dict:

        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat_vars  = [v for v in self.vars if not v.static and v.kind == "categorical"]
        static_cont_vars   = [v for v in self.vars if v.static     and v.kind == "continuous"]
        static_cat_vars    = [v for v in self.vars if v.static     and v.kind == "categorical"]

        ids  = df[self.id_col].unique()
        N, T = len(ids), self.max_len
        nc   = len(temporal_cont_vars)

        Xc        = np.full((N, T, nc), np.nan, dtype=np.float32)
        Xc_mask   = np.zeros((N, T, nc), dtype=np.float32)
        Xcat      = {v.name: np.zeros((N, T), dtype=np.int64)  for v in temporal_cat_vars}
        Xcat_mask = {v.name: np.zeros((N, T), dtype=np.float32) for v in temporal_cat_vars}
        visit_mask  = np.zeros((N, T, 1), dtype=np.float32)
        visit_times_raw = np.zeros((N, T),  dtype=np.float32)   # [v4] tempi assoluti

        sc_mask  = np.ones((N, len(static_cont_vars)), dtype=np.float32)
        scat_mask = {v.name: np.ones(N, dtype=np.float32) for v in static_cat_vars}
        static_rows = []

        for i, pid in enumerate(ids):
            sub      = df[df[self.id_col] == pid].sort_values(self.time_col)
            static_rows.append(sub.iloc[0])
            L        = min(len(sub), T)
            orig_idx = sub.index[:L].values

            visit_mask[i, :L, 0]    = 1.0
            visit_times_raw[i, :L]  = sub[self.time_col].values[:L].astype(np.float32)

            for j, v in enumerate(temporal_cont_vars):
                Xc[i, :L, j]       = sub[v.name].values[:L]
                Xc_mask[i, :L, j]  = cont_masks[v.name][orig_idx]

            for v in temporal_cat_vars:
                Xcat[v.name][i, :L]      = sub[v.name].values[:L]
                Xcat_mask[v.name][i, :L] = cat_masks[v.name][orig_idx]

            first = sub.index[0]
            for j, v in enumerate(static_cont_vars):
                sc_mask[i, j] = cont_masks[v.name][first]
            for v in static_cat_vars:
                scat_mask[v.name][i] = cat_masks[v.name][first]

        return {
            "temporal_cont":      Xc,
            "temporal_cont_mask": Xc_mask,
            "temporal_cat":       Xcat,
            "temporal_cat_mask":  Xcat_mask,
            "visit_mask":         visit_mask,
            "visit_times_raw":    visit_times_raw,   # [v4]
            "df_static":          pd.DataFrame(static_rows).reset_index(drop=True),
            "static_cont_mask":   sc_mask,
            "static_cat_mask":    scat_mask,
        }

    # ==================================================================
    # FIT SCALERS — Z-SCORE (media / deviazione standard)
    # ==================================================================

    def fit_scalers(self, padded: Dict) -> Dict:
        """
        [v5.1] Standard z-score: scaled = (x - μ) / σ.

        SCELTA MOTIVATA:
          Le variabili con distribuzione asimmetrica (BIL, GGT, ALP, AST)
          ricevono log1p PRIMA dello scaling grazie a log_vars nella config.
          Dopo la log-trasformazione le distribuzioni sono quasi simmetriche
          → z-score è appropriato e più semplice del RobustScaler.
          Per le variabili già simmetriche (ALB, ALT, AGE, HEIGHT, WEIGHT)
          lo z-score è la scelta naturale.

        OUTPUT:
          scaled ∈ (-∞, +∞), in pratica ±3σ per quasi tutti i valori.
          Missing/padding → 0.0 (= media normalizzata, valore neutro).

        COMPATIBILITÀ CON SIGMOID:
          Il generatore NON usa Sigmoid sugli output continui (rimossa in v5).
          Gli output lineari possono replicare qualsiasi range z-score.
          L'inverse_transform ricostruisce: x = scaled * σ + μ.

        scalers_cont[v.name] = (mean, std)
        """
        static_cont_vars   = [v for v in self.vars if v.static     and v.kind == "continuous"]
        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]

        # ── Statici ──────────────────────────────────────────────────
        if static_cont_vars and "df_static" in padded:
            cols = []
            for v in static_cont_vars:
                col   = padded["df_static"][v.name].values.astype(np.float32)
                valid = ~np.isnan(col)
                if v.name in self.log_vars:
                    col = col.copy()
                    col[valid] = np.log1p(np.maximum(col[valid], 0))
                mean = float(col[valid].mean()) if valid.any() else 0.0
                std  = float(col[valid].std())  if valid.any() else 1.0
                if std < 1e-8:
                    std = 1.0
                self.scalers_cont[v.name] = (mean, std)
                out = np.where(valid, (col - mean) / std, 0.0)
                cols.append(out[:, None].astype(np.float32))
            if cols:
                padded["static_cont_scaled"] = np.concatenate(cols, axis=1)

        # ── Temporali ────────────────────────────────────────────────
        for j, v in enumerate(temporal_cont_vars):
            data  = padded["temporal_cont"][:, :, j]
            mask  = padded["temporal_cont_mask"][:, :, j]
            valid = (mask == 1) & (~np.isnan(data))
            if v.name in self.log_vars:
                data = data.copy()
                data[valid] = np.log1p(np.maximum(data[valid], 0))
            mean = float(data[valid].mean()) if valid.any() else 0.0
            std  = float(data[valid].std())  if valid.any() else 1.0
            if std < 1e-8:
                std = 1.0
            self.scalers_cont[v.name] = (mean, std)
            padded["temporal_cont"][:, :, j] = np.where(valid, (data - mean) / std, 0.0)

        return padded

    # ==================================================================
    # [v4] NORMALIZE TIME — PER-PAZIENTE
    # ==================================================================

    def normalize_time_per_patient(self, padded: Dict) -> Dict:
        """
        [v4.2] Normalizzazione temporale delta-shift: t_norm[0] = 0 garantito.

        PROBLEMA CON D3_fup O max(visit_times):
          Se la prima visita ha t_raw[0] > 0 (es. arruolamento a 3 mesi
          dall'inizio dello studio), entrambi gli schemi precedenti
          producono t_norm[0] > 0. Il GRU non vede un "inizio" canonico.

        SOLUZIONE — schema delta-shift:
          t_offset[i,t] = t_raw[i,t] - t_raw[i,0]      ≥ 0, offset alla prima visita
          delta_max[i]  = t_offset[i, last_valid]        durata osservata del follow-up
          t_norm[i,t]   = t_offset[i,t] / delta_max[i]  ∈ [0,1], t_norm[0] = 0 ✓

        Livello 2 — durata assoluta:
          followup_norm[i] = delta_max[i] / global_delta_max  ∈ [0,1]
          Cattura "quanto è lungo il follow-up" indipendentemente dall'offset.

        In inverse_transform:
          t_abs[i,t] = t_norm[i,t] × delta_max_i + t_first[i]
          Ma t_first è perso nel processo sintetico → si ricostruisce
          come t_abs[i,t] = t_norm[i,t] × (followup_norm[i] × global_delta_max)
          (relativo all'inizio del follow-up, impostato a 0).

        Salva:
          padded["visit_times"]   : t_norm [N,T] ∈ [0,1], t_norm[0]=0
          padded["followup_norm"] : [N] ∈ [0,1]
          self.global_time_max    : global_delta_max per inverse_transform
        """
        vt = padded["visit_times_raw"]      # [N, T] tempi assoluti
        vm = padded["visit_mask"][:, :, 0]  # [N, T]
        N  = vt.shape[0]

        t_offset    = np.zeros_like(vt)   # tempi relativi alla prima visita
        delta_max   = np.ones(N, dtype=np.float32)  # durata del follow-up

        for i in range(N):
            valid_idx = np.where(vm[i] == 1)[0]
            if len(valid_idx) == 0:
                continue
            t_first = vt[i, valid_idx[0]]     # tempo prima visita
            t_last  = vt[i, valid_idx[-1]]    # tempo ultima visita

            t_offset[i] = vt[i] - t_first     # shift: prima visita → 0
            t_offset[i][vm[i] == 0] = 0.0     # azzera il padding

            d = float(t_last - t_first)
            delta_max[i] = d if d > 1e-8 else 1.0  # pazienti con 1 sola visita: delta=1

        self.global_time_max = float(delta_max.max())
        if self.global_time_max < 1e-8:
            self.global_time_max = 1.0

        # t_norm per-paziente ∈ [0,1], t_norm[:,0] = 0 garantito
        t_norm = np.zeros_like(vt)
        for i in range(N):
            t_norm[i]             = t_offset[i] / delta_max[i]
            t_norm[i][vm[i] == 0] = 0.0

        # followup_norm: durata relativa al massimo globale ∈ [0,1]
        followup_norm = delta_max / self.global_time_max

        padded["visit_times"]   = t_norm.astype(np.float32)
        padded["followup_norm"] = followup_norm.astype(np.float32)
        return padded

    # ==================================================================
    # BUILD STATIC TENSORS
    # ==================================================================

    def build_static_tensors(self, df: pd.DataFrame) -> Dict:
        out = {}
        cat_ohe_parts  = []
        cat_embed_data = {}

        for v in self.vars:
            if not v.static or v.kind != "categorical" or v.name not in df.columns:
                continue
            values  = sorted(v.mapping.values())
            idx_map = {val: i for i, val in enumerate(values)}
            encoded = df[v.name].values

            if v.name in self.embedding_configs:
                indices = np.array([idx_map.get(x, 0) for x in encoded], dtype=np.int64)
                cat_embed_data[v.name] = indices
                if v.name not in self.embeddings:
                    self.embeddings[v.name] = nn.Embedding(
                        len(values), self.embedding_configs[v.name]
                    )
            else:
                indices = np.array([idx_map.get(x, 0) for x in encoded], dtype=np.int64)
                cat_ohe_parts.append(np.eye(len(values), dtype=np.float32)[indices])

        if cat_ohe_parts:
            out["static_cat_ohe"] = np.concatenate(cat_ohe_parts, axis=1)
        if cat_embed_data:
            out["static_cat_embed"] = cat_embed_data
        return out

    # ==================================================================
    # TO TENSORS
    # ==================================================================

    def to_tensors(self, padded: Dict) -> Dict:
        # n_visits: numero di visite valide per paziente [N], calcolato da visit_mask
        # Usato in dgan._train_generator per conditioning del generatore:
        # invece di predire n_v da z_static, si campiona n_visits reale
        # → la distribuzione di lunghezza sequenza è garantita per costruzione.
        n_visits_np = padded["visit_mask"][:, :, 0].sum(axis=1).astype(np.int64)

        out = {
            "temporal_cont":      torch.tensor(padded["temporal_cont"],      dtype=torch.float32),
            "temporal_cont_mask": torch.tensor(padded["temporal_cont_mask"], dtype=torch.float32),
            "visit_mask":         torch.tensor(padded["visit_mask"],         dtype=torch.float32),
            "visit_time":         torch.tensor(padded["visit_times"],        dtype=torch.float32),
            "followup_norm":      torch.tensor(padded["followup_norm"],      dtype=torch.float32),
            "n_visits":           torch.tensor(n_visits_np,                  dtype=torch.long),
            "temporal_cat":       {},
            "temporal_cat_mask":  {},
        }

        if "static_cont_mask" in padded:
            out["static_cont_mask"] = torch.tensor(padded["static_cont_mask"], dtype=torch.float32)

        # static_cat_mask: una maschera per ogni OHE dim
        ohe_mask_parts = []
        for v in self.vars:
            if not v.static or v.kind != "categorical" or v.name in self.embedding_configs:
                continue
            if v.name in padded.get("static_cat_mask", {}):
                n_cat    = len(v.mapping)
                mask_vec = torch.tensor(padded["static_cat_mask"][v.name], dtype=torch.float32)
                ohe_mask_parts.append(mask_vec.unsqueeze(-1).expand(-1, n_cat))
        if ohe_mask_parts:
            out["static_cat_mask"] = torch.cat(ohe_mask_parts, dim=-1)

        # embed mask
        embed_mask = {}
        for v in self.vars:
            if not v.static or v.kind != "categorical" or v.name not in self.embedding_configs:
                continue
            if v.name in padded.get("static_cat_mask", {}):
                embed_mask[v.name] = torch.tensor(
                    padded["static_cat_mask"][v.name], dtype=torch.float32
                )
        if embed_mask:
            out["static_cat_embed_mask"] = embed_mask

        # Temporal categorical → OHE × mask
        for v in [v for v in self.vars if not v.static and v.kind == "categorical"]:
            values  = sorted(v.mapping.values())
            idx_map = {val: i for i, val in enumerate(values)}
            encoded = padded["temporal_cat"][v.name]
            idx     = np.array(
                [[idx_map.get(x, 0) for x in row] for row in encoded],
                dtype=np.int64,
            )
            ohe  = np.eye(len(values), dtype=np.float32)[idx]
            ohe *= padded["temporal_cat_mask"][v.name][:, :, None]
            out["temporal_cat"][v.name]      = torch.tensor(ohe, dtype=torch.float32)
            out["temporal_cat_mask"][v.name] = torch.tensor(
                padded["temporal_cat_mask"][v.name], dtype=torch.float32
            )

        if "static_cont_scaled" in padded:
            out["static_cont"] = torch.tensor(padded["static_cont_scaled"], dtype=torch.float32)
        if "static_cat_ohe" in padded:
            out["static_cat"]  = torch.tensor(padded["static_cat_ohe"], dtype=torch.float32)
        if "static_cat_embed" in padded:
            out["static_cat_embed"] = {
                name: torch.tensor(idx, dtype=torch.long)
                for name, idx in padded["static_cat_embed"].items()
            }

        return out

    # ==================================================================
    # APPLY EMBEDDINGS
    # ==================================================================

    def apply_embeddings(self, batch: Dict) -> torch.Tensor:
        parts = []
        if "static_cont" in batch:
            parts.append(batch["static_cont"])
        if "static_cat" in batch:
            parts.append(batch["static_cat"])
        if "static_cat_embed" in batch:
            for name, indices in batch["static_cat_embed"].items():
                parts.append(self.embeddings[name](indices))
        return torch.cat(parts, dim=-1)

    # ==================================================================
    # INVERSE TRANSFORM
    # ==================================================================

    def inverse_transform(
        self,
        synthetic:         Dict,
        complete_followup: bool = True,
    ) -> pd.DataFrame:
        """
        [v4.2] Ricostruzione tempi con schema delta-shift.

        Poiché t_norm è relativo alla prima visita (t_norm[0]=0 sempre),
        il tempo ricostruito parte da 0 per ogni paziente sintetico:
          delta_max_i = followup_norm[i] × global_time_max
          t_abs[t]    = t_norm[t] × delta_max_i   (prima visita = 0)
          D3_fup      = delta_max_i (aggiunto come colonna)

        ATTENZIONE [BUG PREVENUTO]: questo metodo deve gestire anche
        le variabili con embedding (es. CENTRE) nel blocco 2b.
        Senza il blocco 2b check_data.py lancia KeyError: 'CENTRE'.
        """
        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat_vars  = [v for v in self.vars if not v.static and v.kind == "categorical"]
        static_cont_vars   = [v for v in self.vars if v.static     and v.kind == "continuous"]
        static_cat_vars    = [v for v in self.vars if v.static     and v.kind == "categorical"]

        N, T, _ = synthetic["temporal_cont"].shape
        records  = []

        followup_norm = synthetic.get("followup_norm", None)
        if followup_norm is not None and hasattr(followup_norm, "squeeze"):
            fn = followup_norm
            followup_norm = fn.squeeze(-1) if fn.dim() > 1 else fn  # [N]

        for i in range(N):
            static_data = {}

            # 1. Attributi statici continui
            if "static_cont" in synthetic and synthetic["static_cont"] is not None:
                Z_CLIP = 4.0
                for j, v in enumerate(static_cont_vars):
                    s          = float(synthetic["static_cont"][i, j])
                    s          = max(-Z_CLIP, min(Z_CLIP, s))
                    mean, std  = self.scalers_cont[v.name]
                    val        = s * std + mean
                    if v.name in self.log_vars:
                        val = float(np.expm1(max(val, -30.0)))
                        val = max(0.0, val)
                    static_data[v.name] = val

            # 2a. Attributi statici categorici — OHE [FIX BUG 5: ordine da self.vars]
            if "static_cat" in synthetic and synthetic["static_cat"] is not None:
                offset = 0
                for v in static_cat_vars:
                    if v.name in self.embedding_configs:
                        continue
                    n_cat   = len(v.mapping)
                    oh      = synthetic["static_cat"][i, offset:offset + n_cat]
                    values  = sorted(v.mapping.values())
                    val_enc = values[int(torch.argmax(oh))]
                    static_data[v.name] = self.inverse_maps[v.name][val_enc]
                    offset += n_cat

            # 2b. Attributi statici categorici — EMBEDDING (es. CENTRE)
            # OBBLIGATORIO: senza questo blocco CENTRE manca nel DataFrame
            # e check_data.py lancia KeyError: 'Column not found: CENTRE'.
            if "static_cat_embed_decoded" in synthetic:
                for v in static_cat_vars:
                    if v.name not in self.embedding_configs:
                        continue
                    idx_enc = int(synthetic["static_cat_embed_decoded"][v.name][i])
                    values  = sorted(v.mapping.values())
                    val_enc = values[idx_enc]
                    static_data[v.name] = self.inverse_maps[v.name][val_enc]

            # 3. Durata follow-up sintetico (delta dalla prima visita)
            # Con schema delta-shift: t_abs[0]=0 per costruzione
            if followup_norm is not None:
                delta_max_i = float(followup_norm[i]) * self.global_time_max
            else:
                delta_max_i = self.global_time_max
            static_data["D3_fup"] = delta_max_i

            # 4. Visite temporali [FIX BUG 8]
            valid_steps = [
                t for t in range(T)
                if float(synthetic["visit_mask"][i, t, 0]) > 0.5
            ]

            prev_time = 0.0
            for t in valid_steps:
                if "visit_times" in synthetic and synthetic["visit_times"] is not None:
                    t_norm_val  = min(1.0, max(0.0, float(synthetic["visit_times"][i, t])))
                    time_denorm = t_norm_val * delta_max_i
                else:
                    time_denorm = float(t) * (delta_max_i / max(T - 1, 1))

                delta_t   = max(time_denorm - prev_time, 0.0)
                prev_time = time_denorm

                row = {
                    self.id_col:   f"Synth_{i}",
                    self.time_col: time_denorm,
                    "Delta_t":     delta_t,
                }
                row.update(static_data)

                # Continui temporali
                # [v5.1] Clip z-score a ±4σ prima di inverse_transform.
                # Il generatore produce output lineari non limitati → z-score
                # estremi (es. z=+10) → expm1(z*std+mean) = valori 1e9.
                # ±4σ copre il 99.994% della distribuzione normale e
                # previene overflow nell'inverse_transform senza distorcere
                # i valori nell'intervallo fisicamente plausibile.
                Z_CLIP = 4.0
                for j, v in enumerate(temporal_cont_vars):
                    s          = float(synthetic["temporal_cont"][i, t, j])
                    s          = max(-Z_CLIP, min(Z_CLIP, s))   # clip z-score
                    mean, std  = self.scalers_cont[v.name]
                    val        = s * std + mean
                    if v.name in self.log_vars:
                        val = float(np.expm1(max(val, -30.0)))  # expm1 safe
                        val = max(0.0, val)                      # fisicamente ≥ 0
                    row[v.name] = val

                # Categorici temporali
                for v in temporal_cat_vars:
                    oh      = synthetic["temporal_cat"][v.name][i, t]
                    values  = sorted(v.mapping.values())
                    val_enc = values[int(torch.argmax(oh))]
                    row[v.name] = self.inverse_maps[v.name][val_enc]

                records.append(row)

        df_out = pd.DataFrame(records)
        if len(df_out) > 0:
            df_out = df_out.sort_values(
                [self.id_col, self.time_col]
            ).reset_index(drop=True)
        return df_out

    # ==================================================================
    # DECODE EMBEDDINGS
    # ==================================================================

    def decode_embeddings(
        self, embedded: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        decoded = {}
        for name, vec in embedded.items():
            if name not in self.embeddings:
                continue
            W             = self.embeddings[name].weight.data
            dists         = torch.cdist(vec.float(), W.float(), p=2)
            decoded[name] = torch.argmin(dists, dim=-1)
        return decoded