"""
processing/preprocessor.py  [v6-fully-parametrized]
================================================================================
Rispetto a v5:

  [NUOVO] Gestione warning ed errori espliciti:
    - WARNING se nessuna variabile continua o categorica (temporale o statica)
      → imputazione saltata, non crash silenzioso
    - ERROR se il DataFrame è vuoto o ha meno di 2 pazienti
    - ERROR se la colonna time_col manca
    - ERROR se max_len < 1
    - WARNING se alcuni pazienti hanno 0 visite valide
    - WARNING se max_len viene superato (troncamento silenzioso → ora loggato)
    - WARNING se un paziente ha 1 sola visita (delta_max = 0 → followup forzato)
    - ERROR se la colonna fup_col è dichiarata in config ma mancante nel DataFrame

  [NUOVO] t_FUP dalla colonna reale:
    - Se data_cfg.fup_col è presente nel DataFrame, il follow-up per paziente
      viene letto direttamente da quella colonna (prima visita per paziente).
    - La normalizzazione usa t_FUP come delta_max del paziente invece della
      durata calcolata dall'ultima visita.
    - In questo modo l'ultimo step temporale generato corrisponde esattamente
      al tempo di follow-up del paziente.
    - Se la colonna manca → warning e fallback al comportamento v5
      (delta dalla prima all'ultima visita).

  [NUOVO] Parametri da config:
    - mice_max_iter, knn_neighbors: ora argomenti espliciti del costruttore
    - clip_z: clipping z-score in inverse_transform (da prep_cfg.clip_z)

  Invariato rispetto a v5:
    - MICE per continue temporali e statiche
    - KNNImputer per categoriche
    - valid_flag [N,T] bool
    - Normalizzazione temporale delta-shift per paziente (v4.2)
    - Gestione embedding
================================================================================
"""

import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Optional

from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from config.config_loader import DataConfig

_logger = logging.getLogger(__name__)

MAP_MISSING = "__MISSING__"


class Preprocessor:

    def __init__(
        self,
        data_cfg:          DataConfig,
        embedding_configs: Optional[Dict[str, int]] = None,
        log_vars:          Optional[List[str]]       = None,
        mice_max_iter:     int                       = 10,
        knn_neighbors:     int                       = 5,
        clip_z:            float                     = 4.0,
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
        self.fup_col           = data_cfg.fup_col   # colonna t_FUP dal config
        self.embedding_configs = embedding_configs or {}
        self.embeddings        = nn.ModuleDict()
        self.scalers_cont      = {}
        self.inverse_maps      = {}
        self.global_time_max   = None
        self.mice_max_iter     = mice_max_iter
        self.knn_neighbors     = knn_neighbors
        self.clip_z            = clip_z

        # Imputer fitted durante fit_transform
        self._mice_temporal: Optional[IterativeImputer] = None
        self._mice_static:   Optional[IterativeImputer] = None
        self._knn_static:    Optional[KNNImputer]       = None
        self._knn_temporal:  Optional[KNNImputer]       = None

        self._validate_config()

    # ------------------------------------------------------------------
    def _validate_config(self):
        """Controlla la configurazione al momento della costruzione."""
        if self.max_len < 1:
            raise ValueError(
                f"max_len deve essere >= 1, ricevuto: {self.max_len}"
            )
        if self.mice_max_iter < 1:
            raise ValueError(
                f"mice_max_iter deve essere >= 1, ricevuto: {self.mice_max_iter}"
            )
        if self.knn_neighbors < 1:
            raise ValueError(
                f"knn_neighbors deve essere >= 1, ricevuto: {self.knn_neighbors}"
            )
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
        """
        Trasforma il DataFrame long in tensori pronti per il training.
        Esegue validazione, imputazione, encoding, padding, scaling.
        """
        # ── Validazione DataFrame ──────────────────────────────────────
        self._validate_dataframe(df)

        df            = df.reset_index(drop=True)
        df            = self._force_types(df)
        for v in self.vars:
            if v.kind == "categorical" and v.name in df.columns:
                print(v.name, sorted(df[v.name].unique())[:10])
        df            = self._impute(df)
        df            = self._encode_categoricals(df)
        padded        = self._long_to_padded(df)
        padded        = self._fit_scalers(padded)
        padded        = self._normalize_time_per_patient(padded)
        static_out    = self._build_static_tensors(padded["df_static"])
        padded.update(static_out)
        return self._to_tensors(padded)

    def _validate_dataframe(self, df: pd.DataFrame):
        """Controlla il DataFrame prima dell'elaborazione. Lancia errori/warning."""
        if df is None or len(df) == 0:
            raise ValueError(
                "Il DataFrame è vuoto. Impossibile procedere con il training."
            )

        if self.id_col not in df.columns:
            raise ValueError(
                f"Colonna paziente '{self.id_col}' non trovata nel DataFrame. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        if self.time_col not in df.columns:
            raise ValueError(
                f"Colonna tempo '{self.time_col}' non trovata nel DataFrame. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        n_patients = df[self.id_col].nunique()
        if n_patients < 2:
            raise ValueError(
                f"Il DataFrame contiene solo {n_patients} paziente/i. "
                f"MICE richiede almeno 2 pazienti per funzionare correttamente."
            )

        # Warning se fup_col dichiarata ma mancante
        if self.fup_col and self.fup_col not in df.columns:
            warnings.warn(
                f"La colonna follow-up '{self.fup_col}' non è presente nel DataFrame. "
                f"Il tempo di follow-up verrà calcolato dall'ultima visita registrata. "
                f"Se questo è intenzionale, imposta fup_col=null nel config JSON.",
                UserWarning,
                stacklevel=3,
            )

        # Controlla variabili dichiarate ma mancanti
        all_var_names = [v.name for v in self.vars]
        missing_cols  = [n for n in all_var_names if n not in df.columns]
        if missing_cols:
            warnings.warn(
                f"Le seguenti variabili dichiarate in config NON sono presenti "
                f"nel DataFrame e verranno ignorate: {missing_cols}",
                UserWarning,
                stacklevel=3,
            )

        # Warning se max_len < mediana visite
        visits_per_patient = df.groupby(self.id_col).size()
        median_visits      = int(visits_per_patient.median())
        max_visits         = int(visits_per_patient.max())
        if max_visits > self.max_len:
            n_truncated = int((visits_per_patient > self.max_len).sum())
            warnings.warn(
                f"max_len={self.max_len} è inferiore al numero massimo di visite "
                f"({max_visits}). {n_truncated} pazienti verranno troncati. "
                f"Considera di aumentare max_len nel config.",
                UserWarning,
                stacklevel=3,
            )
        if self.max_len < median_visits:
            warnings.warn(
                f"max_len={self.max_len} è inferiore alla mediana delle visite "
                f"({median_visits}). Più del 50% dei pazienti verrà troncato.",
                UserWarning,
                stacklevel=3,
            )

        # Conta pazienti con 0 visite valide (impossibile, ma per sicurezza)
        zero_visits = (visits_per_patient == 0).sum()
        if zero_visits > 0:
            warnings.warn(
                f"{zero_visits} pazienti hanno 0 visite e verranno ignorati.",
                UserWarning,
                stacklevel=3,
            )

        # Warning variabili temporali assenti per tipo
        temp_cont_names = [v.name for v in self.vars
                           if not v.static and v.kind == "continuous"
                           and v.name in df.columns]
        temp_cat_names  = [v.name for v in self.vars
                           if not v.static and v.kind == "categorical"
                           and v.name in df.columns]
        stat_cont_names = [v.name for v in self.vars
                           if v.static and v.kind == "continuous"
                           and v.name in df.columns]
        stat_cat_names  = [v.name for v in self.vars
                           if v.static and v.kind == "categorical"
                           and v.name in df.columns]

        if not temp_cont_names and not temp_cat_names:
            raise ValueError(
                "Non sono presenti né variabili continue né categoriche temporali. "
                "Il modello richiede almeno un tipo di feature temporale."
            )
        if not temp_cont_names:
            warnings.warn(
                "Nessuna variabile continua temporale trovata nel DataFrame. "
                "MICE temporale verrà saltato. Il discriminatore temporale userà "
                "solo feature categoriche. Verifica la configurazione.",
                UserWarning,
                stacklevel=3,
            )
        if not temp_cat_names:
            warnings.warn(
                "Nessuna variabile categorica temporale trovata nel DataFrame. "
                "KNN temporale verrà saltato.",
                UserWarning,
                stacklevel=3,
            )
        if not stat_cont_names and not stat_cat_names:
            warnings.warn(
                "Nessuna variabile statica trovata nel DataFrame. "
                "Il modello opererà senza features statiche.",
                UserWarning,
                stacklevel=3,
            )
        if not stat_cont_names:
            warnings.warn(
                "Nessuna variabile continua statica trovata nel DataFrame. "
                "MICE statico verrà saltato.",
                UserWarning,
                stacklevel=3,
            )
        if not stat_cat_names:
            warnings.warn(
                "Nessuna variabile categorica statica trovata nel DataFrame. "
                "KNN statico verrà saltato.",
                UserWarning,
                stacklevel=3,
            )

    # ==================================================================
    # FORCE TYPES
    # ==================================================================

    def _force_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df        = df.copy()
        _STR_NANS = {"nan", "none", "<na>", "nat", ""}
        for v in self.vars:
            if v.name not in df.columns:
                continue
            if v.kind == "continuous":
                df[v.name] = pd.to_numeric(df[v.name], errors="coerce")
            #else:
            #    col = df[v.name].astype(str)
            #    col = col.where(~col.str.lower().isin(_STR_NANS), other=MAP_MISSING)
            #    col = col.fillna(MAP_MISSING)
            #    df[v.name] = col
            if v.kind == "categorical":
                col = df[v.name]

                if pd.api.types.is_numeric_dtype(col):
                    # Caso 1: categorica numerica (1.0, 2.0 → "1", "2")
                    col = col.round().astype("Int64").astype(str)
                else:
                    # Caso 2: categorica stringa (es. "PBC0001")
                    col = col.astype(str)

                col = col.where(~col.str.lower().isin(_STR_NANS), other=MAP_MISSING)
                col = col.fillna(MAP_MISSING)

                df[v.name] = col
        return df

    # ==================================================================
    # IMPUTATION — MICE per continue, KNN per categoriche
    # ==================================================================

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputazione in-place sul DataFrame long prima del padding.

        Strategia:
          1. Temporali continue  → MICE (IterativeImputer, mice_max_iter)
          2. Statiche continue   → MICE
          3. Categoriche (stat + temp) → KNNImputer su encoding numerico
          4. Dopo l'imputazione non esistono più NaN/MAP_MISSING.

        Se un gruppo di variabili è assente o ha 0 colonne nel DataFrame,
        il passo corrispondente viene saltato con warning (già emesso in
        _validate_dataframe).
        """
        df = df.copy()

        # ── 1. Temporali continue — interpolazione per paziente ───────
        # Per le variabili temporali, l'interpolazione lineare dentro-paziente
        # e' molto piu' corretta di MICE globale: preserva la traiettoria
        # individuale e non introduce correlazioni artificiose tra pazienti.
        # Strategia per ogni paziente:
        #   a) Interpolazione lineare per i NaN interni alla serie
        #   b) Forward-fill per i NaN iniziali (bordo sinistro)
        #   c) Backward-fill per i NaN finali (bordo destro)
        #   d) Se rimangono NaN (paziente con solo NaN per quella var):
        #      mediana globale della variabile come ultimo fallback
        temp_cont_names = [v.name for v in self.vars
                           if not v.static and v.kind == "continuous"
                           and v.name in df.columns]
        if temp_cont_names:
            has_nan = df[temp_cont_names].isna().any().any()
            if has_nan:
                # Interpola per-paziente mantenendo l'ordine temporale
                df_sorted = df.sort_values([self.id_col, self.time_col])
                for col in temp_cont_names:
                    if not df_sorted[col].isna().any():
                        continue
                    # groupby + interpolate dentro ogni paziente
                    df_sorted[col] = (
                        df_sorted.groupby(self.id_col, group_keys=False)[col]
                        .apply(lambda s: (
                            s.interpolate(method="linear", limit_direction="both")
                             .ffill()
                             .bfill()
                        ))
                    )
                    # Fallback globale per pazienti con colonna interamente NaN
                    if df_sorted[col].isna().any():
                        global_median = df_sorted[col].median()
                        df_sorted[col] = df_sorted[col].fillna(
                            global_median if not pd.isna(global_median) else 0.0
                        )
                # Riordina come il df originale
                df[temp_cont_names] = df_sorted.reindex(df.index)[temp_cont_names]
                n_imputed = df[temp_cont_names].isna().sum().sum()
                if n_imputed > 0:
                    warnings.warn(
                        f"Rimasti {n_imputed} NaN nelle temporali continue dopo interpolazione. "
                        f"Imputati con 0.",
                        UserWarning,
                    )
                    df[temp_cont_names] = df[temp_cont_names].fillna(0.0)

        # ── 2. Statiche continue — MICE ───────────────────────────────
        stat_cont_names = [v.name for v in self.vars
                           if v.static and v.kind == "continuous"
                           and v.name in df.columns]
        if stat_cont_names:
            first_rows = df.groupby(self.id_col, sort=False).first().reset_index()
            sc_data    = first_rows[stat_cont_names].values.astype(np.float64)
            if np.isnan(sc_data).any():
                try:
                    imputer = IterativeImputer(
                        max_iter=self.mice_max_iter, random_state=1, min_value=-np.inf)
                    imputer.fit(sc_data)
                    self._mice_static = imputer
                    df[stat_cont_names] = imputer.transform(
                        df[stat_cont_names].values.astype(np.float64)).astype(np.float32)
                except Exception as e:
                    warnings.warn(
                        f"MICE su variabili statiche continue fallito: {e}. "
                        f"Le colonne con NaN verranno imputate con la mediana.",
                        UserWarning,
                    )
                    for col in stat_cont_names:
                        if df[col].isna().any():
                            df[col] = df[col].fillna(df[col].median())

        # ── 3. Categoriche statiche — KNN ─────────────────────────────
        stat_cat_names = [v.name for v in self.vars
                          if v.static and v.kind == "categorical"
                          and v.name in df.columns]
        if stat_cat_names:
            n_missing = sum((df[n] == MAP_MISSING).sum() for n in stat_cat_names)
            if n_missing > 0:
                try:
                    #df = self._knn_impute_cat(df, stat_cat_names, key="static")
                    for col in stat_cat_names:  # ← for col in, non df[col]!
                        mode = df[col][df[col] != MAP_MISSING].mode()
                        if len(mode) > 0:
                            df[col] = df[col].replace(MAP_MISSING, mode.iloc[0])
                    #df[stat_cat_names] = df[stat_cat_names].fillna(df[col].mode()[0])
                except Exception as e:
                    warnings.warn(
                        f"KNN su categoriche statiche fallito: {e}. "
                        f"I missing verranno sostituiti con la moda.",
                        UserWarning,
                    )
                    for col in stat_cat_names:
                        mode = df[col][df[col] != MAP_MISSING].mode()
                        if len(mode) > 0:
                            df[col] = df[col].replace(MAP_MISSING, mode.iloc[0])

        # ── 4. Categoriche temporali — KNN ────────────────────────────
        temp_cat_names = [v.name for v in self.vars
                          if not v.static and v.kind == "categorical"
                          and v.name in df.columns]
        if temp_cat_names:
            n_missing = sum(
                (df[n] == MAP_MISSING).sum() for n in temp_cat_names)
            if n_missing > 0:
                try:
                    df = self._knn_impute_cat(df, temp_cat_names, key="temporal")
                except Exception as e:
                    warnings.warn(
                        f"KNN su categoriche temporali fallito: {e}. "
                        f"I missing verranno sostituiti con la moda.",
                        UserWarning,
                    )
                    for col in temp_cat_names:
                        mode = df[col][df[col] != MAP_MISSING].mode()
                        if len(mode) > 0:
                            df[col] = df[col].replace(MAP_MISSING, mode.iloc[0])

        return df

    def _knn_impute_cat(
        self, df: pd.DataFrame, cat_names: List[str], key: str
    ) -> pd.DataFrame:
        """
        KNNImputer sulle categoriche: MAP_MISSING → NaN numerico → imputa → arrotonda.
        Mappa i valori di stringa in interi prima (0=missing), poi ritorna a stringa.
        """
        if len(cat_names) == 0:
            return df

        num_df = pd.DataFrame(index=df.index)

        for name in cat_names:
            col  = df[name]
            uniq = [v for v in col.unique() if v != MAP_MISSING]
            if not uniq:
                warnings.warn(
                    f"Variabile categorica '{name}': tutti i valori sono missing. "
                    f"Impossibile imputare con KNN. La variabile verrà ignorata.",
                    UserWarning,
                )
                num_df[name] = np.nan
                continue
            code = {u: i + 1 for i, u in enumerate(uniq)}
            code[MAP_MISSING] = np.nan
            num_df[name] = col.map(lambda x, c=code: c.get(x, np.nan))

        # Controlla se ha senso fare KNN (almeno 1 valore non-NaN per colonna)
        all_nan_cols = [c for c in num_df.columns if num_df[c].isna().all()]
        if all_nan_cols:
            warnings.warn(
                f"Colonne con tutti NaN nel KNN categorico: {all_nan_cols}. "
                f"Verranno lasciate come NaN e gestite dal fallback.",
                UserWarning,
            )
            num_df = num_df.drop(columns=all_nan_cols)
            cat_names = [n for n in cat_names if n not in all_nan_cols]

        if len(cat_names) == 0:
            return df

        n_neighbors = min(self.knn_neighbors, len(df) - 1)
        if n_neighbors < 1:
            n_neighbors = 1

        imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        arr = imp.fit_transform(num_df.values.astype(np.float64))

        if key == "static":
            self._knn_static = imp
        else:
            self._knn_temporal = imp

        arr_rounded = np.round(arr).astype(int).clip(1, None)
        for j, name in enumerate(cat_names):
            codes = arr_rounded[:, j]
            #inv   = {v: k for k, v in v.mapping.items()}  #{i + 1: u for i, u in enumerate([v for v in df[name].unique() if v != MAP_MISSING])}
            #if not inv:
            #    continue
            #df[name] = [inv.get(c, inv[min(inv)]) for c in codes]
            # Recuperiamo l'oggetto variabile corretto per questa colonna
            var_cfg = next((v for v in self.vars if v.name == name), None)
            
            if var_cfg is None or not var_cfg.mapping:
                continue
                
            # Creiamo la mappa inversa: codice numerico -> stringa originale
            inv = {v_int: k_str for k_str, v_int in var_cfg.mapping.items()}
            
            # Applichiamo la decodifica
            df[name] = [inv.get(c, inv[min(inv.keys())]) for c in codes]

        return df

    # ==================================================================
    # ENCODE CATEGORICALS
    # ==================================================================

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dopo l'imputazione non ci sono più MAP_MISSING → encoding diretto."""
        df = df.copy()
        for v in self.vars:
            if v.kind != "categorical" or v.name not in df.columns:
                continue
            # Controlla valori fuori mapping
            known = set(v.mapping.keys())
            unknown = set(df[v.name].unique()) - known - {MAP_MISSING}
            if unknown:
                warnings.warn(
                    f"Variable '{v.name}': valori non nel mapping: {unknown}. "
                    f"Verranno mappati al codice 1 (primo valore del mapping).",
                    UserWarning,
                )
            self.inverse_maps[v.name] = {val: key for key, val in v.mapping.items()}
            df[v.name] = df[v.name].map(
                lambda x, m=v.mapping: m.get(x, 1)
            ).astype(int)
        return df

    # ==================================================================
    # LONG → PADDED  [valid_flag invece di visit_mask]
    # ==================================================================

    def _long_to_padded(self, df: pd.DataFrame) -> Dict:
        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat_vars  = [v for v in self.vars if not v.static and v.kind == "categorical"]

        ids  = df[self.id_col].unique()
        N, T = len(ids), self.max_len
        nc   = len(temporal_cont_vars)

        Xc              = np.zeros((N, T, nc),  dtype=np.float32)
        Xcat            = {v.name: np.zeros((N, T), dtype=np.int64) for v in temporal_cat_vars}
        valid_flag      = np.zeros((N, T),       dtype=bool)
        visit_times_raw = np.zeros((N, T),       dtype=np.float32)
        fup_times       = np.zeros(N,            dtype=np.float32)  # t_FUP per paziente
        static_rows     = []

        # Controlla se la colonna fup_col è disponibile
        has_fup_col = bool(self.fup_col and self.fup_col in df.columns)

        for i, pid in enumerate(ids):
            sub = df[df[self.id_col] == pid].sort_values(self.time_col)
            if len(sub) == 0:
                _logger.debug("Paziente '%s' ha 0 visite. Trattato come padding.", pid)
                static_rows.append(df.iloc[0])  # placeholder
                continue

            static_rows.append(sub.iloc[0])
            L = min(len(sub), T)

            if len(sub) > T:
                # Usa debug invece di UserWarning per evitare spam
                # (il warning aggregato viene emesso in _validate_dataframe)
                _logger.debug(
                    "Paziente '%s' ha %d visite > max_len=%d. Troncato.",
                    pid, len(sub), T,
                )

            valid_flag[i, :L]      = True
            visit_times_raw[i, :L] = sub[self.time_col].values[:L].astype(np.float32)

            # t_FUP reale dal DataFrame
            if has_fup_col:
                fup_val = sub[self.fup_col].iloc[0]
                if pd.isna(fup_val):
                    warnings.warn(
                        f"Paziente '{pid}': colonna '{self.fup_col}' è NaN. "
                        f"Uso l'ultima visita come follow-up.",
                        UserWarning,
                    )
                    fup_times[i] = float(sub[self.time_col].values[L - 1])
                else:
                    fup_times[i] = float(fup_val)
                    # Sanity check: t_FUP >= ultima visita registrata
                    last_visit = float(sub[self.time_col].values[L - 1])
                    if fup_times[i] < last_visit - 1e-6:
                        warnings.warn(
                            f"Paziente '{pid}': t_FUP={fup_times[i]:.2f} è "
                            f"inferiore all'ultima visita ({last_visit:.2f}). "
                            f"Uso l'ultima visita come follow-up.",
                            UserWarning,
                        )
                        fup_times[i] = last_visit
            else:
                # Fallback: usa l'ultima visita
                fup_times[i] = float(sub[self.time_col].values[L - 1])

            for j, v in enumerate(temporal_cont_vars):
                if v.name in sub.columns:
                    Xc[i, :L, j] = sub[v.name].values[:L]

            for v in temporal_cat_vars:
                if v.name in sub.columns:
                    Xcat[v.name][i, :L] = sub[v.name].values[:L]

        return {
            "temporal_cont":   Xc,
            "temporal_cat":    Xcat,
            "valid_flag":      valid_flag,
            "visit_times_raw": visit_times_raw,
            "fup_times":       fup_times,       # ← t_FUP reale per paziente [N]
            "df_static":       pd.DataFrame(static_rows).reset_index(drop=True),
        }

    # ==================================================================
    # FIT SCALERS — Z-SCORE
    # ==================================================================

    def _fit_scalers(self, padded: Dict) -> Dict:
        static_cont_vars   = [v for v in self.vars if v.static     and v.kind == "continuous"]
        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        vf                 = padded["valid_flag"]  # [N,T] bool

        # ── Statici ───────────────────────────────────────────────────
        if static_cont_vars and "df_static" in padded:
            cols = []
            for v in static_cont_vars:
                if v.name not in padded["df_static"].columns:
                    continue
                col = padded["df_static"][v.name].values.astype(np.float32)
                if v.name in self.log_vars:
                    col = np.log1p(np.maximum(col, 0))
                mean = float(col.mean()) if len(col) > 0 else 0.0
                std  = float(col.std())  if len(col) > 0 else 1.0
                if std < 1e-8:
                    std = 1.0
                self.scalers_cont[v.name] = (mean, std)
                cols.append(((col - mean) / std)[:, None].astype(np.float32))
            if cols:
                padded["static_cont_scaled"] = np.concatenate(cols, axis=1)

        # ── Temporali ─────────────────────────────────────────────────
        for j, v in enumerate(temporal_cont_vars):
            data  = padded["temporal_cont"][:, :, j]
            if v.name in self.log_vars:
                data = data.copy()
                data[vf] = np.log1p(np.maximum(data[vf], 0))
            mean = float(data[vf].mean()) if vf.any() else 0.0
            std  = float(data[vf].std())  if vf.any() else 1.0
            if std < 1e-8:
                std = 1.0
            self.scalers_cont[v.name] = (mean, std)
            scaled = np.where(vf, (data - mean) / std, 0.0)
            padded["temporal_cont"][:, :, j] = scaled

        return padded

    # ==================================================================
    # NORMALIZE TIME — PER-PAZIENTE con t_FUP reale
    # ==================================================================

    def _normalize_time_per_patient(self, padded: Dict) -> Dict:
        """
        Normalizzazione temporale per paziente.

        Se fup_times[i] > 0 (da colonna t_FUP reale):
          - delta_max[i] = fup_times[i] - t_first[i]
          - t_norm[i, t] = (t_raw - t_first) / delta_max  ∈ [0, 1]
          - L'ultimo step valido corrisponde esattamente a t_FUP.

        Se fup_times[i] == 0 o la colonna non c'era:
          - Fallback: delta_max[i] = t_last - t_first (comportamento v5)
        """
        vt  = padded["visit_times_raw"]   # [N,T]
        vf  = padded["valid_flag"]        # [N,T] bool
        fup = padded["fup_times"]         # [N]  (0 se non disponibile)
        N   = vt.shape[0]

        t_offset  = np.zeros_like(vt)
        delta_max = np.ones(N, dtype=np.float32)
        self.global_time_max = float(delta_max.max())

        single_visit_warned = False

        for i in range(N):
            valid_idx = np.where(vf[i])[0]
            if len(valid_idx) == 0:
                continue

            t_first = vt[i, valid_idx[0]]

            # Determina delta_max: usa t_FUP se disponibile e > 0
            if fup[i] > 1e-8:
                # t_FUP è assoluto → delta = t_FUP - t_first
                d = float(fup[i]) - float(t_first)
                if d < 1e-8:
                    # t_FUP coincide o precede la prima visita (non dovrebbe accadere)
                    if not single_visit_warned:
                        warnings.warn(
                            f"Paziente indice {i}: delta t_FUP - t_first = {d:.4f} "
                            f"(quasi zero). Potrebbe indicare t_FUP espresso come "
                            f"durata dal baseline invece che come data assoluta. "
                            f"Usa delta_max = max(fup[i], ultima_visita - t_first).",
                            UserWarning,
                        )
                        single_visit_warned = True
                    d = max(float(vt[i, valid_idx[-1]]) - float(t_first), 1.0)
                delta_max[i] = d
            else:
                # Fallback: distanza prima-ultima visita
                t_last = vt[i, valid_idx[-1]]
                d      = float(t_last) - float(t_first)
                if d < 1e-8:
                    if not single_visit_warned:
                        warnings.warn(
                            f"Paziente indice {i} ha una sola visita o visite "
                            f"con lo stesso timestamp. delta_max forzato a 1.0.",
                            UserWarning,
                        )
                        single_visit_warned = True
                    d = 1.0
                delta_max[i] = d

            t_offset[i]          = vt[i] - t_first
            t_offset[i][~vf[i]]  = 0.0

        self.global_time_max = float(delta_max.max())
        if self.global_time_max < 1e-8:
            self.global_time_max = 1.0

        t_norm        = np.zeros_like(vt)
        followup_norm = np.zeros(N, dtype=np.float32)

        for i in range(N):
            # Normalizzazione per-paziente: visit_times ∈ [0, 1] dove 1.0 = t_FUP
            t_norm[i]         = t_offset[i] / delta_max[i]
            t_norm[i][~vf[i]]  = 0.0
            followup_norm[i]   = delta_max[i] / self.global_time_max

        padded["visit_times"]   = t_norm.astype(np.float32)
        padded["followup_norm"] = followup_norm.astype(np.float32)
        # Salva delta_max per uso in inverse_transform
        padded["delta_max_per_patient"] = delta_max
        return padded

    # ==================================================================
    # BUILD STATIC TENSORS
    # ==================================================================

    def _build_static_tensors(self, df: pd.DataFrame) -> Dict:
        out            = {}
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
                        len(values), self.embedding_configs[v.name])
            else:
                indices = np.array([idx_map.get(x, 0) for x in encoded], dtype=np.int64)
                cat_ohe_parts.append(np.eye(len(values), dtype=np.float32)[indices])

        if cat_ohe_parts:
            out["static_cat_ohe"]   = np.concatenate(cat_ohe_parts, axis=1)
        if cat_embed_data:
            out["static_cat_embed"] = cat_embed_data
        return out

    # ==================================================================
    # TO TENSORS
    # ==================================================================

    def _to_tensors(self, padded: Dict) -> Dict:
        vf_np       = padded["valid_flag"]   # bool [N,T]
        n_visits_np = vf_np.sum(axis=1).astype(np.int64)

        out = {
            "temporal_cont":  torch.tensor(padded["temporal_cont"], dtype=torch.float32),
            "valid_flag":     torch.tensor(vf_np,                   dtype=torch.bool),
            "visit_time":     torch.tensor(padded["visit_times"],    dtype=torch.float32),
            "followup_norm":  torch.tensor(padded["followup_norm"],  dtype=torch.float32),
            "n_visits":       torch.tensor(n_visits_np,              dtype=torch.long),
            "temporal_cat":   {},
        }

        # Temporal categorical → OHE (padding = zero vector, non usato)
        for v in [v for v in self.vars if not v.static and v.kind == "categorical"]:
            values  = sorted(v.mapping.values())
            idx_map = {val: i for i, val in enumerate(values)}
            encoded = padded["temporal_cat"][v.name]
            idx     = np.array(
                [[idx_map.get(x, 0) for x in row] for row in encoded],
                dtype=np.int64,
            )
            ohe                        = np.eye(len(values), dtype=np.float32)[idx]
            out["temporal_cat"][v.name] = torch.tensor(ohe, dtype=torch.float32)

        if "static_cont_scaled" in padded:
            out["static_cont"] = torch.tensor(padded["static_cont_scaled"], dtype=torch.float32)
        if "static_cat_ohe" in padded:
            out["static_cat"]  = torch.tensor(padded["static_cat_ohe"],     dtype=torch.float32)
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
        if not parts:
            raise ValueError(
                "Nessuna feature statica disponibile per apply_embeddings. "
                "Verifica che static_cont, static_cat o static_cat_embed siano presenti nel batch."
            )
        return torch.cat(parts, dim=-1)

    # ==================================================================
    # INVERSE TRANSFORM
    # ==================================================================

    def inverse_transform(self, synthetic: Dict) -> pd.DataFrame:
        """
        Ricostruzione dal formato sintetico a DataFrame long.
        valid_flag determina gli step attivi.
        Il tempo dell'ultimo step viene agganciato a followup_norm * global_time_max.
        """
        temporal_cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat_vars  = [v for v in self.vars if not v.static and v.kind == "categorical"]
        static_cont_vars   = [v for v in self.vars if v.static     and v.kind == "continuous"]
        static_cat_vars    = [v for v in self.vars if v.static     and v.kind == "categorical"]

        tc   = synthetic["temporal_cont"]
        N, T = tc.shape[0], tc.shape[1]
        records = []

        if "valid_flag" in synthetic:
            vf_raw = synthetic["valid_flag"]
        elif "visit_mask" in synthetic:
            vf_raw = synthetic["visit_mask"]
        else:
            vf_raw = torch.ones(N, T, dtype=torch.bool)

        if vf_raw.dim() == 3:
            vf_raw = vf_raw.squeeze(-1)

        followup_norm = synthetic.get("followup_norm", None)
        if followup_norm is not None and followup_norm.dim() > 1:
            followup_norm = followup_norm.squeeze(-1)

        Z_CLIP = self.clip_z   # da config, non hardcoded

        for i in range(N):
            static_data = {}

            # 1. Statici continui
            if "static_cont" in synthetic and synthetic["static_cont"] is not None:
                for j, v in enumerate(static_cont_vars):
                    s         = float(synthetic["static_cont"][i, j])
                    s         = max(-Z_CLIP, min(Z_CLIP, s))
                    mean, std = self.scalers_cont[v.name]
                    val       = s * std + mean
                    if v.name in self.log_vars:
                        val = float(np.expm1(max(val, -30.0)))
                        val = max(0.0, val)
                    static_data[v.name] = val

            # 2. Statici categorici — OHE
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

            # 3. Statici categorici — Embedding
            if "static_cat_embed_decoded" in synthetic:
                for v in static_cat_vars:
                    if v.name not in self.embedding_configs:
                        continue
                    idx_enc = int(synthetic["static_cat_embed_decoded"][v.name][i])
                    values  = sorted(v.mapping.values())
                    val_enc = values[idx_enc]
                    static_data[v.name] = self.inverse_maps[v.name][val_enc]

            # 4. Durata follow-up (denormalizzata)
            if followup_norm is not None:
                delta_max_i = float(followup_norm[i]) * self.global_time_max
            else:
                delta_max_i = self.global_time_max
            static_data[self.fup_col] = delta_max_i   # colonna t_FUP configurabile

            # 5. Visite temporali — usa valid_flag
            if vf_raw.dtype == torch.bool:
                valid_steps = vf_raw[i].nonzero(as_tuple=True)[0].tolist()
            else:
                valid_steps = (vf_raw[i] > 0.5).nonzero(as_tuple=True)[0].tolist()

            n_valid = len(valid_steps)
            prev_time = 0.0
            for step_i, t in enumerate(valid_steps):
                if "visit_times" in synthetic and synthetic["visit_times"] is not None:
                    t_norm_val = min(1.0, max(0.0, float(synthetic["visit_times"][i, t])))
                    # Aggancia l'ultimo step esattamente a delta_max_i
                    # Normalizzazione per-paziente: tutti gli step (incluso l'ultimo)
                    # usano delta_max_i = followup_norm * global_time_max del paziente.
                    # L'ultimo step avrà time_denorm ≈ delta_max_i = t_FUP.
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

                for j, v in enumerate(temporal_cont_vars):
                    s         = float(tc[i, t, j])
                    s         = max(-Z_CLIP, min(Z_CLIP, s))
                    mean, std = self.scalers_cont[v.name]
                    val       = s * std + mean
                    if v.name in self.log_vars:
                        val = float(np.expm1(max(val, -30.0)))
                        val = max(0.0, val)
                    row[v.name] = val

                for v in temporal_cat_vars:
                    oh      = synthetic["temporal_cat"][v.name][i, t]
                    values  = sorted(v.mapping.values())
                    val_enc = values[int(torch.argmax(oh))]
                    row[v.name] = self.inverse_maps[v.name][val_enc]

                records.append(row)

        df_out = pd.DataFrame(records)
        if len(df_out) > 0:
            df_out = df_out.sort_values(
                [self.id_col, self.time_col]).reset_index(drop=True)
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
                warnings.warn(
                    f"Embedding '{name}' non trovato nel preprocessore. "
                    f"Verrà saltato nel decode.",
                    UserWarning,
                )
                continue
            W             = self.embeddings[name].weight.data
            dists         = torch.cdist(vec.float(), W.float(), p=2)
            decoded[name] = torch.argmin(dists, dim=-1)
        return decoded