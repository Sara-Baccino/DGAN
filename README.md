**1. Panoramica**

DoppelGANger (DGAN) è un modello generativo basato su GAN progettato per produrre dati longitudinali sintetici con caratteristiche statiche (baseline) e temporali (follow-up).
L’obiettivo principale è generare sequenze realistiche di variabili cliniche o altre serie temporali di pazienti, preservando le relazioni tra le variabili statiche e temporali, anche quando alcune variabili sono irreversibili (ad esempio eventi clinici che una volta accaduti non possono tornare indietro).

Il modello supporta anche Differential Privacy tramite Opacus, permettendo di generare dati sintetici rispettando la privacy dei pazienti reali.

**2. Struttura del modello**

Il modello è composto da tre componenti principali:
  - Generatore Gerarchico;
  - Discriminatore Statico;
  - Discriminatore Temporale


**2.1 Generatore Gerarchico (Hierarchical Generator)**

Il generatore crea sequenze sintetiche sia per le variabili statiche sia per quelle temporali:

Statiche (baseline):

  -Variabili continue → passano attraverso una rete fully connected con ReLU.
  
  -Variabili categoriche → codificate tramite softmax (con Gumbel-Softmax per campionamento differenziabile).
  
  -Variabili irreversibili → inizializzate come probabilità e aggiornate tramite una logica di hazard nel tempo.


Temporali (follow-up):

Generazione step-by-step tramite un GRU, dove l’input ad ogni step include:

  -Rumore latente temporale (z_temporal)
  
  -Embedding delle feature statiche
  
  -Tempo normalizzato [0,1]

  -Stato delle variabili irreversibili


Head separate per:

  -Variabili continue → sigmoid scaling
  
  -Variabili categoriche → Gumbel-Softmax
  
  -Variabili irreversibili → aggiornamento tramite hazard + teacher forcing opzionale

Il mask viene applicato solo al livello di output per mantenere consistenza tra lunghezze diverse delle sequenze.

**2.2 Discriminatori**

Per addestrare il generatore, DGAN utilizza due discriminatori separati, entrambi basati su WGAN con gradient penalty:

*Static Discriminator:*

  Rete fully connected a più layer con LeakyReLU.
  
  Valuta la plausibilità delle feature statiche sintetiche rispetto a quelle reali.


*Temporal Discriminator:*

  GRU per processare sequenze temporali.
  
  Combina la rappresentazione temporale con le feature statiche.
  
  Applica masking per lunghezze diverse delle sequenze.

Output finale → plausibilità della sequenza completa.

**3. Funzionamento del Training**

Il training segue la logica tipica delle Wasserstein GAN con gradient penalty, con alcune modifiche per le sequenze temporali e le variabili irreversibili:

*Input:*

Batch di dati reali pre-elaborati dal preprocessor.

Rumore latente z_static e z_temporal campionato da distribuzione normale.


*Aggiornamento discriminatori:*

Per ogni batch, il generatore produce dati sintetici.


*Calcolo delle loss WGAN:*

d_real vs d_fake per static e temporal discriminator.

Applicazione del gradient penalty per stabilizzare il training.

Backprop e aggiornamento dei pesi dei discriminatori.

Ripetizione per più round per stabilizzare il training dei discriminatori rispetto al generatore.


*Aggiornamento generatore:*

Generazione di dati sintetici usando z_static e z_temporal.

Loss del generatore: massimizzare il punteggio dei discriminatori (-(D_static + D_temporal)).

Se presenti variabili irreversibili, viene calcolata anche la irreversibility loss, penalizzando i flip 1→0 nelle sequenze binarie.


*Annealing della temperatura Gumbel-Softmax:*

La temperatura τ parte da un valore iniziale e decresce esponenzialmente fino ad un valore minimo, per rendere le variabili categoriali quasi discrete alla fine del training.

**Differential Privacy (opzionale):**

Se attivato, Opacus aggiunge rumore ai gradienti dei discriminatori e limita la norma dei gradienti.

Durante il training, viene monitorato ε per valutare la privacy.


*Logging e tracciamento:*

Loss generator e discriminatori salvati per ogni batch.

Monitoraggio di ε se DP è attivo.

Possibilità di generare plot dell’andamento delle loss nel tempo.

**4. Generazione dei dati sintetici**

Una volta addestrato, il modello genera dati sintetici con generate(n_samples):

- Campiona rumore latente z_static e z_temporal.

- Passa attraverso il generatore gerarchico.

- Applica inverse transform per riportare i valori scalati e le one-hot categorical a valori interpretabili.

- Restituisce dataset sintetico completo con:

    -Feature statiche continue e categoriche
    -Sequenze temporali continue e categoriali
    -Possibilità di generare sequenze complete o tronche (simulando mancati follow-up)

**5. Output del training**

Dopo il training, il modello fornisce:

DGAN.generate() → dataset sintetico completo

*File salvati:*

Modello PyTorch (.pt)

Plot delle loss (generator, discriminatori, ε)

Dataset sintetico in formato Excel o CSV





Teacher forcing: permette di utilizzare i valori reali delle variabili irreversibili durante la generazione per stabilizzare il training.

Hazard irreversible: gestisce le variabili irreversibili step-by-step, calcolando la probabilità di evento e aggiornando lo stato binario.

Mask temporale: gestisce sequenze di lunghezza variabile, ma non è più necessario fornire un value-mask al generatore.


