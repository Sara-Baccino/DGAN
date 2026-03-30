**1. Panoramica**

DoppelGANger (DGAN) è un modello generativo basato su GAN progettato per produrre dati longitudinali sintetici con caratteristiche statiche (baseline) e temporali (follow-up).
L’obiettivo principale è generare sequenze realistiche di variabili cliniche o altre serie temporali di pazienti, preservando le relazioni tra le variabili statiche e temporali.

Il modello può supportare anche Differential Privacy tramite Opacus, permettendo di generare dati sintetici rispettando la privacy dei pazienti reali.

**2. Struttura del modello**

Il modello è composto da tre componenti principali:
  - Generatore Gerarchico
  - Discriminatore Statico
  - Discriminatore Temporale


**2.1 Generatore Gerarchico (Hierarchical Generator)**

Il generatore crea sequenze sintetiche sia per le variabili statiche sia per quelle temporali:

**Statiche** (baseline): generate a partire dalla componente di rumore z_static  [B, z_s]. In particolare, vengono gestit 3 tipi di variabili:

  -Variabili continue → passano attraverso una rete fully connected.
  -Variabili categoriche → codificate tramite softmax (con Gumbel-Softmax per campionamento differenziabile).
  -Variabili irreversibili → inizializzate come probabilità e aggiornate tramite una logica di hazard nel tempo.

La fase statica funziona da inizializzazione per quella temporale. Infatti, il rumore $z_{static}$ passa attraverso 3 heads lineari:
  -to_h0: Si crea lo stato interno iniziale h0 della rete GRU e rappresenta le informazioni di baseline del paziente;
  -followup_head: Predice quanto durerà in totale la storia clinica del paziente ($t_{FUP}$), indipendentemente da quante visite farà;
  -visit_times: Predice quante visite avrà quel paziente.

**Temporali** (follow-up): generate a partire dalla componente di rumore z_temporal [B, T, z_t]. 
Se configurato con noise_ar_rho > 0, questo rumore segue un processo AR(1), ovvero il rumore allo step $t$ è correlato a quello dello step $t-1$.

La generazione temporale è autoregressiva step-by-step e avviene tramite GRU, eseguendo un loop sul range(T). L’input ad ogni step include:

  -Rumore latente temporale specifico per lo step corrente  (z_temporal, t)
  -Valori delle analisi generate allo step precedente
  -Intervallo di tempo passato dall'ultima visita (delta_prev)

Una volta che la GRU ha aggiornato il suo stato interno $h_t$, il modello usa diverse teste specializzate per trasformare quel vettore astratto in dati clinici:

  -Variabili continue (temporal_cont_head)
  -Intervalli di Tempo (interval_head), viene usato Softplus per garantire che delta_t>0
  -Variabili categoriche, tramite Gumbel-Softmax
  -Variabili irreversibili, tramite cummax su hazard

Il tempo delle visite viene normalizzato in [0,1], dove 1 non coincide necessariamente con t_FUP e si ottiene con cumsum(delta).
Il numero di visite è limitato da min_visits e max_visits.
Viene applicata una valid_flag (maschera booleana) per mantenere consistenza tra lunghezze diverse delle sequenze, indicando quali visite sono di padding e quali sono reali.


**2.2 Discriminatori**

Per addestrare il generatore, DGAN utilizza due discriminatori separati, basati su WGAN con gradient penalty. 
Entrambi i modelli integrano la Spectral Normalization per garantire la stabilità del training e il controllo della costante di Lipschitz.

*Static Discriminator*: Valuta la plausibilità delle feature statiche sintetiche rispetto a quelle reali.
Rete MLP Residua che prende in input un vettore concatenato $[B, D_{static}]$ contenente variabili continue, categoriche (one-hot) ed embedding.
Include teste di classificazione multi-task che forzano il modello a ricostruire le classi originali delle variabili statiche, riducendo il rischio di mode collapse.  


*Temporal Discriminator*: Analizza la dinamica delle visite nel tempo, integrando il contesto statico per validare la coerenza clinica.

Input: Tensore 3D $[B, T, D_{temp} + 1]$ dove $+1$ rappresenta il followup_norm (tempo totale di osservazione) broadcastato su ogni step

Supporta due modalità configurabili tramite il parametro "arch":
-CNN (Default): Rete neurale convoluzionale dilatata. Utilizza campi ricettivi esponenziali per catturare dipendenze a lungo termine senza l'instabilità delle reti ricorrenti.
-GRU: Unità ricorrente per processare la sequenza in modo autoregressivo (per retrocompatibilità).

Meccanismo di Attention Pooling guidato dal valid_flag. Il discriminatore calcola uno score di importanza per ogni visita e ignora attivamente i passi di padding, permettendo di gestire coorti con numero di visite altamente variabile.


**3. Funzionamento del Training**

Il modello viene addestrato seguendo lo schema WGAN-GP (Wasserstein GAN con Gradient Penalty), ottimizzato per la stabilità su dataset medici di piccole dimensioni.

#### Meccanismi di Stabilità
- **Feature Matching (FM):** Oltre alla loss avversaria standard, il Generatore minimizza la distanza MSE tra le attivazioni intermedie del Discriminatore per dati reali e sintetici. 
Questo forza la corrispondenza delle distribuzioni latenti profonde.
- **GP Curriculum:** Il coefficiente di Gradient Penalty viene aumentato linearmente durante le prime epoche per permettere una fase di esplorazione iniziale senza instabilità dei gradienti.
- **EMA (Exponential Moving Average):** Viene mantenuta una copia "slow" dei pesi del generatore (EMA Generator) per la fase di inferenza, riducendo le oscillazioni tipiche delle GAN.

#### Loss Funzionali Specifiche
Il Generatore ottimizza una funzione di costo multi-obiettivo:
1.  **Adversarial Loss:** WGAN-GP per la verosimiglianza globale.
2.  **Variance Loss:** Penalizza discrepanze nella deviazione standard delle feature continue (fondamentale per biomarker come l'albumina).
3.  **Delta/Interval Loss:** Assicura che la distribuzione temporale delle visite (intervalli Δt) rispecchi la frequenza clinica reale.
4.  **Autocorrelation Loss:** Forza il modello a rispettare la "smoothness" temporale (es. impedisce salti bio-fisicamente impossibili tra due step adiacenti).
5.  **Irreversibility Loss:** Vincola variabili come il decesso o la cirrosi a stati non regressivi.
