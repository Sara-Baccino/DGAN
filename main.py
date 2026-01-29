import pandas as pd
from config.config_loader import load_config, build_data_config
from processing.processor import Preprocessor
from utils.check_data import *
from model.dgan import DGAN
from utils.check_data import check_one_hot, check_no_nan


def main():
    # LOAD CONFIG
    config_path = "config/data_config.json"
    
    time_cfg, variables, model_cfg = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)



    # LOAD DATA
    dataset_name ="PBC_UDCA_long.xlsx"
    df = pd.read_excel(dataset_name)

    # --- LOGICA DI VALIDAZIONE VISITE ---
    # Calcoliamo il numero massimo di visite per paziente nei dati reali
    actual_max_visits = df.groupby(time_cfg.patient_id).size().max()

    if time_cfg.max_visits > actual_max_visits:
        raise ValueError(
            f"ERRORE: La config richiede {time_cfg.max_visits} visite, "
            f"ma il paziente con più dati ne ha solo {actual_max_visits}. "
            f"Riduci 'max_visits' nel JSON."
        )
    
    elif time_cfg.max_visits < actual_max_visits:
        print(f"[INFO] Troncamento in corso: i dati hanno fino a {actual_max_visits} visite, "
            f"ma la config ne prevede {time_cfg.max_visits}. Rimuovo le eccedenze.")
        
        # Tronchiamo i dati mantenendo solo le prime N visite per ogni paziente
        df = df.sort_values([time_cfg.patient_id, time_cfg.visit_column])
        df = df.groupby(time_cfg.patient_id).head(time_cfg.max_visits).reset_index(drop=True)

    print("Preprocessing...")
    pre = Preprocessor(data_cfg)
    data = pre.fit_transform(df)

    print("Preprocessing OK.\n")
    print("Formato tensori: \n")
    for k, v in data.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"{k}/{kk}: {vv.shape}")
        else:
            print(k, v.shape)

    # ===============================
    # CONSISTENCY CHECKS
    # ===============================
    print("Check consistenza: \n")

    check_no_nan(data)

    for v in data["temporal_cat"].values():
        check_one_hot(v.numpy())

    print("\nInizializzazione modello...")

    model = DGAN(time_cfg=time_cfg, variables=variables, model_cfg=model_cfg)

    print("Training...")
    model.fit(data)

    print("Generazione dati sintetici (follow-up completi)...")

    synthetic = model.generate(
        n_samples=len(data["static_cont"]),
        temperature=0.1,          # quasi deterministico
        return_torch=False
    )

    print("Inverse transform...")
    synthetic_df = pre.inverse_transform(
        synthetic,
        complete_followup=True
    )

    synthetic_df.to_excel("synthetic_complete_followup.xlsx", index=False)

    print("Dataset sintetico salvato.")



if __name__ == "__main__":
    main()
