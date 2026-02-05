import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from config.config_loader import DataConfig

MAP_MISSING = "__MISSING__"


class Preprocessor:
    def __init__(self, data_cfg: DataConfig, embedding_configs: Optional[Dict[str, int]] = None):
        """
        Args:
            data_cfg: configurazione delle variabili
            embedding_configs: dict {var_name: embedding_dim} per variabili categoriche
                              da rappresentare con embedding invece di OHE
                              es. {"CENTRE": 6} per ridurre 48 categorie a 6 dimensioni
        """
        self.vars = (
            data_cfg.static_cont +
            data_cfg.static_cat +
            data_cfg.temporal_cont +
            data_cfg.temporal_cat
        )

        self.max_len        = data_cfg.max_len
        self.id_col         = data_cfg.patient_id_col
        self.time_col       = data_cfg.time_col

        # configurazione embedding
        self.embedding_configs = embedding_configs or {}
        self.embeddings        = nn.ModuleDict()

        # scalers e mappature
        self.scalers_cont  = {}   # {var_name: (min, max)}
        self.inverse_maps  = {}   # {var_name: {encoded_int: original_str}}

        # normalizzazione tempo globale
        self.global_time_max = None


    # ======================================================
    # FIT + TRANSFORM
    # ======================================================
    def fit_transform(self, df: pd.DataFrame) -> Dict:
        """
        Pipeline completa:
        1. reset_index  → allinea indici numpy e pandas (fix BUG A)
        2. force types
        3. encode categoricals  → maschere missing
        4. process continuous   → maschere missing
        5. long → padded        (con df_static separato)
        6. fit scalers          → scrive padded["static_cont_scaled"],
                                  normalizza temporal_cont in-place
        7. normalize time       → visit_times / global_max
        8. build static cat     → OHE + embedding indices (fix BUG B: non tocca static_cont)
        9. to_tensors
        """
        # FIX BUG A: index 0..N-1 garantito → maschere numpy e sub.index allineati
        df = df.reset_index(drop=True)

        df               = self.force_types(df)
        df, cat_masks    = self.encode_categoricals(df)
        df, cont_masks   = self.process_continuous(df)

        padded           = self.long_to_padded(df, cat_masks, cont_masks)
        padded           = self.fit_scalers(padded)
        padded           = self.normalize_time_global(padded)

        # FIX BUG B: build_static_tensors produce solo categorici;
        # static_cont_scaled vive già in padded dopo fit_scalers
        static_cat_out   = self.build_static_tensors(padded["df_static"])
        padded.update(static_cat_out)

        return self.to_tensors(padded)


    # ======================================================
    # FORCE TYPES
    # ======================================================
    def force_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for v in self.vars:
            if v.name not in df.columns:
                continue
            if v.kind == "continuous":
                df[v.name] = pd.to_numeric(df[v.name], errors="coerce")
            else:
               # df[v.name] = df[v.name].fillna(MAP_MISSING).astype(str)
               df[v.name] = df[v.name].astype(str)
        return df


    # ======================================================
    # ENCODE CATEGORICALS
    # ======================================================
    def encode_categoricals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categoricals → integer.  Maschere: 1 = presente, 0 = missing."""
        df    = df.copy()
        masks = {}

        for v in self.vars:
            if v.kind != "categorical" or v.name not in df.columns:
                continue

            mapping = v.mapping
            self.inverse_maps[v.name] = {val: key for key, val in mapping.items()}

            col = df[v.name]
            masks[v.name] = (col != MAP_MISSING).astype(float).values
            # chiude su mapping nella lambda per evitare late-binding
            #df[v.name] = col.map(lambda x, m=mapping: m.get(x, 0)).astype(int)
            df[v.name] = col.map(lambda x, m=mapping: m[x] if x in m else 0).astype(int)


        return df, masks


    # ======================================================
    # CONTINUOUS
    # ======================================================
    def process_continuous(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Maschere: 1 = presente, 0 = missing.  NaN rimangono NaN."""
        df    = df.copy()
        masks = {}

        for v in self.vars:
            if v.kind != "continuous" or v.name not in df.columns:
                continue
            masks[v.name] = (~df[v.name].isna()).astype(float).values

        return df, masks


    # ======================================================
    # LONG → PADDED
    # ======================================================
    def long_to_padded(
        self,
        df: pd.DataFrame,
        cat_masks:  Dict[str, np.ndarray],
        cont_masks: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Converte long → padded.
        PRECONDIZIONE: df.index è 0..len(df)-1 (da reset_index in fit_transform).
        """
        temporal_cont_names = [v.name for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat_vars   = [v        for v in self.vars if not v.static and v.kind == "categorical"]
        static_cont_vars    = [v        for v in self.vars if v.static  and v.kind == "continuous"]
        static_cat_vars     = [v        for v in self.vars if v.static  and v.kind == "categorical"]

        ids    = df[self.id_col].unique()
        N, T   = len(ids), self.max_len

        # ---- tensori temporali ----
        Xc          = np.full((N, T, len(temporal_cont_names)), np.nan, dtype=np.float32)
        Xcat        = {v.name: np.zeros((N, T), dtype=np.int64) for v in temporal_cat_vars}
        Xc_mask     = np.zeros((N, T, len(temporal_cont_names)), dtype=np.float32)
        Xcat_mask   = {v.name: np.zeros((N, T), dtype=np.float32) for v in temporal_cat_vars}
        visit_mask  = np.zeros((N, T, 1), dtype=np.float32)
        visit_times = np.zeros((N, T),    dtype=np.float32)

        # ---- maschere statiche (una riga per paziente) ----
        static_cont_mask     = np.ones((N, len(static_cont_vars)), dtype=np.float32)
        static_cat_mask_dict = {v.name: np.ones(N, dtype=np.float32) for v in static_cat_vars}

        static_rows = []

        for i, pid in enumerate(ids):
            sub = df[df[self.id_col] == pid].sort_values(self.time_col)
            static_rows.append(sub.iloc[0])

            L = min(len(sub), T)
            visit_mask[i, :L, 0] = 1.0
            visit_times[i, :L]   = sub[self.time_col].values[:L].astype(np.float32)

            # indici posizionali nel df originale (dopo reset_index = label == posizione)
            orig_idx = sub.index[:L].values          # array di interi

            # continuous temporali
            for j, col in enumerate(temporal_cont_names):
                Xc[i, :L, j]      = sub[col].values[:L]
                Xc_mask[i, :L, j] = cont_masks[col][orig_idx]

            # categorical temporali
            for v in temporal_cat_vars:
                Xcat[v.name][i, :L]      = sub[v.name].values[:L]
                Xcat_mask[v.name][i, :L] = cat_masks[v.name][orig_idx]

            # maschere statiche dalla prima visita
            first = sub.index[0]
            for j, v in enumerate(static_cont_vars):
                static_cont_mask[i, j] = cont_masks[v.name][first]
            for v in static_cat_vars:
                static_cat_mask_dict[v.name][i] = cat_masks[v.name][first]

        return {
            "temporal_cont":      Xc,
            "temporal_cat":       Xcat,
            "temporal_cont_mask": Xc_mask,
            "temporal_cat_mask":  Xcat_mask,
            "visit_mask":         visit_mask,
            "visit_times":        visit_times,
            "df_static":          pd.DataFrame(static_rows).reset_index(drop=True),
            "static_cont_mask":   static_cont_mask,
            "static_cat_mask":    static_cat_mask_dict,
        }


    # ======================================================
    # FIT SCALERS (MIN-MAX)
    # ======================================================
    def fit_scalers(self, padded: Dict) -> Dict:
        """
        Min-max [0,1] solo su valori presenti.  NaN → 0 dopo scaling.
        Scrive padded["static_cont_scaled"].
        """
        # --- static continuous ---
        static_cont_vars = [v for v in self.vars if v.kind == "continuous" and v.static]
        if static_cont_vars and "df_static" in padded:
            cols = []
            for v in static_cont_vars:
                col   = padded["df_static"][v.name].values.astype(np.float32)
                valid = ~np.isnan(col)

                minv = float(col[valid].min()) if valid.any() else 0.0
                maxv = float(col[valid].max()) if valid.any() else 1.0
                self.scalers_cont[v.name] = (minv, maxv)

                cols.append(
                    np.where(valid, (col - minv) / (maxv - minv + 1e-8), 0.0)[:, None]
                )
            if cols:
                padded["static_cont_scaled"] = np.concatenate(cols, axis=1).astype(np.float32)

        # --- temporal continuous (in-place) ---
        temporal_cont_vars = [v for v in self.vars if v.kind == "continuous" and not v.static]
        for j, v in enumerate(temporal_cont_vars):
            data  = padded["temporal_cont"][:, :, j]
            mask  = padded["temporal_cont_mask"][:, :, j]
            valid = (mask == 1) & (~np.isnan(data))

            minv = float(data[valid].min()) if valid.any() else 0.0
            maxv = float(data[valid].max()) if valid.any() else 1.0
            self.scalers_cont[v.name] = (minv, maxv)

            padded["temporal_cont"][:, :, j] = np.where(
                valid, (data - minv) / (maxv - minv + 1e-8), 0.0
            )

        return padded


    # ======================================================
    # NORMALIZE TIME GLOBALLY
    # ======================================================
    def normalize_time_global(self, padded: Dict) -> Dict:
        vt   = padded["visit_times"]
        mask = padded["visit_mask"][:, :, 0]
        valid = vt[mask == 1]

        self.global_time_max = float(valid.max()) if len(valid) > 0 else 1.0
        padded["visit_times"] = vt / (self.global_time_max + 1e-8)
        return padded


    # ======================================================
    # BUILD STATIC TENSORS (solo categorici)
    # ======================================================
    def build_static_tensors(self, df: pd.DataFrame) -> Dict:
        """
        Produce solo componenti categoriche statiche:
        - OHE per variabili senza embedding
        - indici grezzi (0-based densi) per variabili con embedding
        Le continuous sono già in padded["static_cont_scaled"].
        
        IMPORTANTE: i missing (encoding=0) non sono nel mapping.
        Usa idx_map.get(x, 0) con default=0 per gestirli.
        """
        out            = {}
        cat_ohe        = []
        cat_embed_data = {}

        for v in self.vars:
            if not v.static or v.kind != "categorical" or v.name not in df.columns:
                continue

            # Mappa: encoded_value → indice denso 0-based
            # NOTA: il mapping NON contiene 0 (riservato ai missing)
            values  = sorted(v.mapping.values())
            idx_map = {val: i for i, val in enumerate(values)}

            # Prendi i valori encoded dal dataframe
            encoded_values = df[v.name].values  # possono contenere 0 (missing)

            if v.name in self.embedding_configs:
                # Per gli embedding: usa get(x, 0) → default 0 se missing
                indices = np.array([idx_map.get(x, 0) for x in encoded_values], dtype=np.int64)
                cat_embed_data[v.name] = indices

                if v.name not in self.embeddings:
                    self.embeddings[v.name] = nn.Embedding(len(values), self.embedding_configs[v.name])
            else:
                # Per OHE: usa get(x, 0) → default 0 se missing
                indices = np.array([idx_map.get(x, 0) for x in encoded_values], dtype=np.int64)
                cat_ohe.append(np.eye(len(values), dtype=np.float32)[indices])

        if cat_ohe:
            out["static_cat_ohe"] = np.concatenate(cat_ohe, axis=1)
        if cat_embed_data:
            out["static_cat_embed"] = cat_embed_data

        return out


    # ======================================================
    # TO TENSORS
    # ======================================================
    def to_tensors(self, padded: Dict) -> Dict:
        """Conversione finale a torch tensors."""
        out = {
            "temporal_cont":      torch.tensor(padded["temporal_cont"],      dtype=torch.float32),
            "temporal_cont_mask": torch.tensor(padded["temporal_cont_mask"], dtype=torch.float32),
            "visit_mask":         torch.tensor(padded["visit_mask"],         dtype=torch.float32),
            "visit_time":         torch.tensor(padded["visit_times"],        dtype=torch.float32),
            "temporal_cat":       {},
            "temporal_cat_mask":  {},
        }

        # --- maschere statiche continuous ---
        if "static_cont_mask" in padded:
            out["static_cont_mask"] = torch.tensor(padded["static_cont_mask"], dtype=torch.float32)

        # --- maschere statiche categoriche OHE (espanse per ogni classe) ---
        ohe_mask_parts = []
        for v in self.vars:
            if not v.static or v.kind != "categorical" or v.name in self.embedding_configs:
                continue
            if v.name in padded.get("static_cat_mask", {}):
                n_cat     = len(v.mapping)
                mask_vec  = torch.tensor(padded["static_cat_mask"][v.name], dtype=torch.float32)
                ohe_mask_parts.append(mask_vec.unsqueeze(-1).expand(-1, n_cat))
        if ohe_mask_parts:
            out["static_cat_mask"] = torch.cat(ohe_mask_parts, dim=-1)

        # --- maschere statiche embedding ---
        embed_mask = {}
        for v in self.vars:
            if not v.static or v.kind != "categorical" or v.name not in self.embedding_configs:
                continue
            if v.name in padded.get("static_cat_mask", {}):
                embed_mask[v.name] = torch.tensor(padded["static_cat_mask"][v.name], dtype=torch.float32)
        if embed_mask:
            out["static_cat_embed_mask"] = embed_mask

        # --- temporal categorical: OHE per variabile ---
        for v in [v for v in self.vars if not v.static and v.kind == "categorical"]:
            values  = sorted(v.mapping.values())
            idx_map = {val: i for i, val in enumerate(values)}
            
            # Missing (encoding=0) non sono nel mapping → usa get con default=0
            # Poi la maschera azzererà: [1,0] * mask=0 → [0,0]
            encoded_vals = padded["temporal_cat"][v.name]  # [N, T]
            idx = np.array([[idx_map.get(x, 0) for x in row] for row in encoded_vals], dtype=np.int64)
            
            ohe = np.eye(len(values), dtype=np.float32)[idx]  # [N, T, K]
            
            # Applica maschera
            mask_expanded = padded["temporal_cat_mask"][v.name][:, :, None]
            ohe = ohe * mask_expanded

            out["temporal_cat"][v.name]      = torch.tensor(ohe, dtype=torch.float32)
            out["temporal_cat_mask"][v.name] = torch.tensor(
                padded["temporal_cat_mask"][v.name], dtype=torch.float32
            )

        # --- static continuous (già scalati) ---
        if "static_cont_scaled" in padded:
            out["static_cont"] = torch.tensor(padded["static_cont_scaled"], dtype=torch.float32)

        # --- static categorical OHE ---
        if "static_cat_ohe" in padded:
            out["static_cat"] = torch.tensor(padded["static_cat_ohe"], dtype=torch.float32)

        # --- static categorical embedding indices ---
        if "static_cat_embed" in padded:
            out["static_cat_embed"] = {
                name: torch.tensor(idx, dtype=torch.long)
                for name, idx in padded["static_cat_embed"].items()
            }

        return out


    # ======================================================
    # APPLY EMBEDDINGS  (helper per discriminatore)
    # ======================================================
    def apply_embeddings(self, batch: Dict) -> torch.Tensor:
        """Concatena tutte le componenti statiche → [B, static_dim_total]."""
        parts = []
        if "static_cont" in batch:
            parts.append(batch["static_cont"])
        if "static_cat" in batch:
            parts.append(batch["static_cat"])
        if "static_cat_embed" in batch:
            for name, indices in batch["static_cat_embed"].items():
                parts.append(self.embeddings[name](indices))
        return torch.cat(parts, dim=-1)


    # ======================================================
    # INVERSE TRANSFORM
    # ======================================================
    def inverse_transform(
        self,
        synthetic: Dict,
        complete_followup: bool = True
    ) -> pd.DataFrame:
        """
        Ricostruisce DataFrame long da tensori sintetici.

        Output format:
            - Una riga per visita
            - Variabili statiche ripetute
            - Colonna Delta_t (tempo dalla visita precedente; 0 per la prima)
            - Nessun missing value

        synthetic deve contenere:
            temporal_cont       : [N, T, n_cont]
            temporal_cat        : {name: [N, T, n_cat]}   ← dict per variabile
            visit_mask          : [N, T, 1]
            visit_times         : [N, T]                  ← normalizzati [0,1]
            static_cont         : [N, n_static_cont]      (opz.)
            static_cat          : [N, sum_ohe]            (opz.)
            static_cat_embed_decoded : {name: [N]}        (opz.)
        """
        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat_vars  = [v for v in self.vars if not v.static and v.kind == "categorical"]
        static_cont_vars   = [v for v in self.vars if v.static  and v.kind == "continuous"]
        static_cat_vars    = [v for v in self.vars if v.static  and v.kind == "categorical"]

        N, T, _ = synthetic["temporal_cont"].shape
        records  = []

        for i in range(N):
            # -------- static (comune a tutte le visite) --------
            static_data = {}

            if "static_cont" in synthetic and synthetic["static_cont"] is not None:
                for j, v in enumerate(static_cont_vars):
                    s = float(synthetic["static_cont"][i, j])
                    minv, maxv = self.scalers_cont[v.name]
                    static_data[v.name] = s * (maxv - minv + 1e-8) + minv

            if "static_cat" in synthetic and synthetic["static_cat"] is not None:
                offset = 0
                for v in static_cat_vars:
                    if v.name in self.embedding_configs:
                        continue
                    n_cat        = len(v.mapping)
                    oh           = synthetic["static_cat"][i, offset:offset + n_cat]
                    values       = sorted(v.mapping.values())
                    val_enc      = values[int(torch.argmax(oh))]
                    static_data[v.name] = self.inverse_maps[v.name][val_enc]
                    offset += n_cat

            if "static_cat_embed_decoded" in synthetic:
                for v in static_cat_vars:
                    if v.name not in self.embedding_configs:
                        continue
                    idx_enc      = int(synthetic["static_cat_embed_decoded"][v.name][i])
                    values       = sorted(v.mapping.values())
                    val_enc      = values[idx_enc]
                    static_data[v.name] = self.inverse_maps[v.name][val_enc]

            # -------- temporal --------
            prev_time = 0.0

            for t in range(T):
                if float(synthetic["visit_mask"][i, t, 0]) == 0 and not complete_followup:
                    continue

                # FIX BUG C: denormalizza visit_times generati, non l'indice t
                if "visit_times" in synthetic and synthetic["visit_times"] is not None:
                    time_norm   = float(synthetic["visit_times"][i, t])
                    time_denorm = time_norm * (self.global_time_max + 1e-8)
                else:
                    time_denorm = t * (self.global_time_max / max(T - 1, 1))

                # Delta_t
                delta_t   = max(time_denorm - prev_time, 0.0) if t > 0 else 0.0
                prev_time = time_denorm

                row = {
                    self.id_col:   f"Synth_{i}",
                    self.time_col: time_denorm,
                    "Delta_t":     delta_t,
                }
                row.update(static_data)

                # continuous temporali
                for j, v in enumerate(temporal_cont_vars):
                    s          = float(synthetic["temporal_cont"][i, t, j])
                    minv, maxv = self.scalers_cont[v.name]
                    row[v.name] = s * (maxv - minv + 1e-8) + minv

                # categorical temporali  ← dict per variabile
                for v in temporal_cat_vars:
                    oh       = synthetic["temporal_cat"][v.name][i, t]
                    values   = sorted(v.mapping.values())
                    val_enc  = values[int(torch.argmax(oh))]
                    row[v.name] = self.inverse_maps[v.name][val_enc]

                records.append(row)

        df_out = pd.DataFrame(records)
        if len(df_out) > 0:
            df_out = df_out.sort_values([self.id_col, self.time_col]).reset_index(drop=True)
        return df_out


    # ======================================================
    # DECODE EMBEDDINGS
    # ======================================================
    def decode_embeddings(self, embedded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Nearest-neighbor sui pesi del layer embedding → indici categorici.
        Args:  {var_name: [N, embed_dim]}
        Out:   {var_name: [N]}
        """
        decoded = {}
        for name, vec in embedded.items():
            if name not in self.embeddings:
                continue
            W         = self.embeddings[name].weight.data          # [K, D]
            dists     = torch.cdist(vec.float(), W.float(), p=2)   # [N, K]
            decoded[name] = torch.argmin(dists, dim=-1)
        return decoded