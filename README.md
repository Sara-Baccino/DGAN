**1. Panoramica**

DoppelGANger (DGAN) è un modello generativo basato su GAN progettato per produrre dataset tabulari longitudinali sintetici con caratteristiche statiche (baseline) e temporali (follow-up), con righe multiple per paziente.
L’obiettivo principale è generare sequenze realistiche di variabili cliniche o altre serie temporali di pazienti, preservando le relazioni tra le variabili statiche e temporali.

Il modello può supportare anche Differential Privacy tramite Opacus, permettendo di generare dati sintetici rispettando la privacy dei pazienti reali.

**2. Struttura del modello**

Il modello è composto da tre componenti principali:
  - Generatore Gerarchico
  - Discriminatore Statico
  - Discriminatore Temporale

Il Generatore genera prima le variabili statiche a partire da rumore z_static e le utilizza per condizionare la generazione delle feature temporali a partire da z_temporal.

Il discriminatore statico ha un'architettura prettamente lineare e produce uno score osservando le variabili statiche dei dati.

L'architettura della parte temporale del Generatore e del Discriminatore sono configurabili da json (lstm, cnn, gru).


**3. Preprocessing**

Il dataset reale di input viene elaborato in modo da estrarre variabili statiche, temporali e irreversibili tramite lettura config.
I dati vengono imputati e convertiti in tensori.
Le sequenze temporali possono essere troncate ad una lunghezza massima max_length. 
Viene eseguito un padding per i tensori dei pazienti che hanno meno di max_length visite e una valid_flag indica quali sono le visite effettive.
La colonna che nel json viene indicata come visit_column è quella che contiene lo stato di avanzamento temporale reale del paziente.
Vengono calcolati gli intervalli di tempo tra due visite consecutive e viene indicato il tempo del paziente normalizzato sul suo tempo di visita e sul tempo di visita massimo del dataset.

**4. Funzionamento del Training**

Il modello viene addestrato seguendo lo schema WGAN-GP (Wasserstein GAN con Gradient Penalty), ottimizzato per la stabilità su dataset medici di piccole dimensioni.

Alla w-gan loss del generatore si sommano delle componenti di penalizzazione per garantire realismo per quanto riguarda la struttura temporale delle visite e dei valori registrati.
