# Development Log — ECG Arrhythmia Classification with 1D CNN

Documento di lavoro (in italiano, per studio personale) aggiornato progressivamente durante lo sviluppo. Contiene le scelte fatte, il perché, e il come. Per la presentazione pubblica del progetto vedi il `README.md` della repository.

**Nota sullo scope**: questo progetto è stato scollegato dalla componente hardware/embedded originariamente prevista (STM32 + MAX30003). Il modello qui sviluppato resta un esercizio metodologico a sé stante su classificazione ECG con dati scarsi/sbilanciati. L'eventuale integrazione futura in un dispositivo embedded sarà un progetto separato, che potrà riusare o meno questo modello.

**Ultimo aggiornamento**: PROGETTO DI MODELLAZIONE CONCLUSO — Tentativo 8 (pipeline gerarchica + soglie ottimizzate) fallito per overfitting nella calibrazione soglie per classe (test macro F1 = 0.36, peggiore del progetto). **Risultato ufficiale confermato: Tentativo 6/7, macro F1 sul test = 0.4488**. Violazione della regola "test set una volta sola" riconosciuta esplicitamente (secondo utilizzo).

---

## 1. Obiettivo del progetto

Esplorare la classificazione di pattern patologici del battito cardiaco (aritmie) su segnale ECG grezzo (raw waveform), tramite una rete neurale convoluzionale 1D (CNN), senza feature engineering manuale — con particolare attenzione alla gestione metodologica di scarsità di dati e forte sbilanciamento di classe.

**Contesto**: primo progetto pratico con le reti neurali. Obiettivo: costruire un portfolio che dimostri comprensione di processo, metodo e capacità diagnostica di fronte a un problema reale di dati scarsi/sbilanciati — non solo un numero di accuratezza finale.

---

## 2. Pipeline complessiva del progetto

```
[1] Colab: esplorazione dataset pubblico (MIT-BIH)
[2] Colab: preprocessing segnale (filtri, segmentazione, normalizzazione)
[3] Colab: costruzione e training CNN 1D (TensorFlow/Keras)
[4] Colab: valutazione (accuracy, confusion matrix, split per paziente)
[5] Colab: conversione + quantizzazione INT8 (Keras → TFLite)
[6] Download modello .tflite (locale)
[7] Locale: STM32CubeIDE + X-CUBE-AI → genera codice C dal modello
[8] Locale: firmware completo (driver SPI MAX30003 + preprocessing + inferenza)
[9] Flash su STM32F411 via ST-Link
```

**Stato attuale**: Blocco [1] in corso.

---

## 4. Scelte metodologiche chiave (e perché)

### 4.1 — Niente feature engineering manuale
Input alla rete: segnale grezzo (raw waveform) dopo solo preprocessing di segnale (filtraggio, normalizzazione, segmentazione). Nessuna feature calcolata a mano (durata QRS, ampiezza, rapporti tra onde, ecc.) — la CNN deve imparare da sola le caratteristiche morfologiche rilevanti.

**Motivazione**: obiettivo di apprendimento end-to-end; le CNN 1D su raw ECG generalizzano meglio su pattern morfologici sottili rispetto a feature ingegnerizzate, secondo la letteratura corrente.

### 4.2 — R-peak detection: software, non hardware
Nonostante il MAX30003 offra R-peak detection hardware integrata, si è scelto di implementare la detection in software (Pan-Tompkins semplificato) sul segnale grezzo letto dal FIFO.

**Motivazione**: mantenere coerenza con l'approccio "pattern recognition puro" — la segmentazione del battito è un'operazione di segnale necessaria (serve per allineare le finestre), non una feature diagnostica. Delegare la detection all'hardware avrebbe comunque richiesto validazione della sua robustezza su morfologie anomale (es. PVC).

### 4.3 — Framework: TensorFlow/Keras
**Motivazione**: percorso di deploy diretto verso microcontrollori Cortex-M4 tramite TensorFlow Lite for Microcontrollers e X-CUBE-AI (tool ufficiale ST), senza passaggi di conversione intermedi (es. ONNX) necessari con altri framework come PyTorch.

### 4.4 — Training su Google Colab
**Motivazione**: GPU gratuita, zero setup locale, output finale del training è un file piccolo (.tflite quantizzato) facilmente scaricabile per il deploy successivo in locale.

### 4.5 — Standard di classificazione: AAMI
Le classi di battito vengono raggruppate secondo lo standard AAMI (5 macro-classi) invece di usare direttamente i simboli di annotazione MIT-BIH.

**Motivazione**: alcuni simboli di annotazione hanno pochissimi esempi nel dataset (es. `S`: 2 occorrenze, `e`: 16, `Q`: 33 — insufficienti per training/validazione affidabili). Il raggruppamento AAMI è inoltre lo standard usato nella letteratura scientifica su MIT-BIH, rendendo i risultati confrontabili.

### 4.6 — Split dataset per paziente (pianificato, non ancora implementato)
Il train/validation/test split dovrà essere stratificato per paziente (record), non per singolo battito, per evitare data leakage (la rete non deve "imparare a riconoscere il paziente" invece della patologia).

### 4.7 — Parametri di segmentazione del battito
Finestra fissa asimmetrica attorno al picco R: **100 campioni prima, 200 dopo** (totale 300 campioni, ~0.83 secondi a 360Hz). Asimmetria scelta perché il tratto ST e l'onda T (dopo il picco R) contengono più informazione diagnostica del solo tratto pre-QRS.

**Canale usato**: solo canale 0 di ogni record (derivazione MLII nella maggior parte dei casi) — coerente con un device single-lead come il MAX30003.

**Battiti scartati**: solo quelli troppo vicini all'inizio/fine del record da non poter riempire la finestra (48 su 109.494 totali, circa 1 per record).

### 4.8 — Preprocessing del segnale
Applicato **prima** della segmentazione, sull'intero segnale di ogni record (non sulla singola finestra già tagliata), perché il filtro necessita di contesto ai bordi per funzionare correttamente:

- **Filtro passa-banda Butterworth** (0.5–40 Hz, ordine 4, `filtfilt` per fase zero): rimuove baseline wander e rumore ad alta frequenza
- **Normalizzazione z-score per singola finestra** (dopo la segmentazione): ogni segmento viene centrato e scalato individualmente, per compensare la variabilità di ampiezza tra record diversi

Confermato sperimentalmente: media ≈ 0, std ≈ 1 su ogni segmento normalizzato.

**Bug noto e risolto**: nella prima verifica visiva, il codice recuperava erroneamente il primo battito del record invece del battito effettivamente corrispondente al segmento in esame — causava un errore di indicizzazione negativa silenzioso (wraparound Python) quando il primo battito era troppo vicino all'inizio della registrazione. Risolto tracciando esplicitamente la posizione originale di ogni battito (`beat_position`) durante la segmentazione.

### 4.9 — Split Train/Validation/Test
Split **per record** (paziente), non per singolo battito, per evitare data leakage. Proporzioni: 70% train / 15% validation / 15% test, sui 48 record totali.

**Risultato dello split**:

| Set | N. record | N. segmenti |
|---|---|---|
| Training | 33 | 75.395 |
| Validation | 7 | 14.911 |
| Test | 8 | 19.140 |

**Distribuzione classi nel training set** (confrontata con la distribuzione complessiva):

| Classe | % Training | % Dataset completo |
|---|---|---|
| N | 81.4% | 82.8% |
| Q | 7.9% | 7.3% |
| V | 7.0% | 6.6% |
| S | 3.1% | 2.5% |
| F | 0.6% | 0.7% |

Distribuzione coerente tra training set e dataset completo — lo split casuale per record non introduce squilibri aggiuntivi significativi rispetto allo sbilanciamento strutturale già noto (sezione 5.5).


---

### 4.10 — Gestione sbilanciamento di classe
Tecnica scelta: **class weights** (non oversampling), calcolati con `sklearn.utils.class_weight.compute_class_weight(class_weight='balanced')` sul training set.

**Motivazione**: i class weights pesano l'errore sulle classi minoritarie nella loss function senza duplicare fisicamente esempi, riducendo il rischio di overfitting sulle classi rare rispetto all'oversampling — scelta più sicura per un primo modello.

**Pesi calcolati**:

| Classe | Peso |
|---|---|
| F | 35.732 |
| S | 6.455 |
| Q | 2.525 |
| V | 2.855 |
| N | 0.246 |

Il peso elevato per F (quasi 36x) riflette la sua rarità nel dataset (0.6% del training set) — da monitorare in fase di training per eventuale instabilità della loss.

### 4.11 — Architettura CNN 1D
Rete compatta a 3 blocchi convoluzionali, pensata per il deploy su STM32F411:

```
Input (300, 1)
→ Conv1D(16, kernel=9) + BatchNorm + ReLU + MaxPool(2)
→ Conv1D(32, kernel=7) + BatchNorm + ReLU + MaxPool(2)
→ Conv1D(64, kernel=5) + BatchNorm + ReLU
→ GlobalAveragePooling1D
→ Dense(5, softmax)
```

**Risultato**: **14.853 parametri totali (58.02 KB)**, di cui 14.629 addestrabili. Ampiamente sotto il target di 50k parametri stabilito in fase di dimensionamento hardware (sezione "Stima concreta per la tua architettura" nelle note di progetto) — margine ulteriore disponibile per eventuale espansione della rete se l'accuratezza richiedesse più capacità. Con quantizzazione INT8 attesa una riduzione di ulteriori ~4x.

### 4.12 — Errori riscontrati e correzioni adottate

Tracciamento degli errori reali incontrati durante lo sviluppo e delle decisioni prese per risolverli — utile come diario tecnico e come dimostrazione del processo iterativo di debug in un progetto ML.

**Errore 1 — Bug di indicizzazione nella verifica visiva del preprocessing.**
Nel confronto "prima/dopo preprocessing", il codice recuperava il primo battito assoluto del record (`beat_samples[0]`) invece del battito effettivamente corrispondente al segmento in esame. Quando il primo battito era vicino all'inizio della registrazione, la sottrazione della finestra produceva un indice negativo che Python interpreta come accesso dalla fine dell'array (wraparound), risultando in un plot vuoto/errato.
*Correzione*: tracciata esplicitamente la posizione originale (`beat_position`) di ogni battito durante la segmentazione, usata per recuperare correttamente il segmento grezzo corrispondente.

**Errore 2 — Training instabile con class weights estremi.**
Primo training con class weights bilanciati (peso F = 35.7x) mostra forte instabilità: val_accuracy oscilla tra 0.16 e 0.91 tra epoche successive, senza convergenza pulita. Il modello risultante (ripristinato dall'epoca migliore per val_loss) ha macro avg F1 = 0.36: le classi minoritarie F e S restano sostanzialmente non apprese (F1 ≈ 0.01-0.02) nonostante il peso elevato assegnato loro.
*Causa probabile*: il peso 35x amplifica enormemente il gradiente ogni volta che un raro esempio di classe F compare in un mini-batch, causando aggiornamenti troppo bruschi, aggravati da un learning rate di default non adatto a questo scenario.
*Correzione adottata*: vedi sezione 4.13.

**Osservazione 3 — Classe Q assente nel validation set (non un bug di codice).**
Il classification report mostra 0 esempi di classe Q nel validation set. Verificato che è un effetto reale dello split per record: i battiti Q (prevalentemente da pacemaker) sono concentrati in pochi record specifici di MIT-BIH, nessuno dei quali è stato assegnato al fold di validation nello split casuale eseguito. Segnala un limite dello split per paziente quando applicato a classi fortemente concentrate in pochi soggetti — da monitorare anche sul test set.

**Errore 4 — Capping dei class weights insufficiente: instabilità persistente e classi minoritarie non apprese.**
Dopo il capping (peso massimo 10x) e la riduzione del learning rate (0.0003) con `ReduceLROnPlateau`, il training sui dati risulta pulito (loss di training monotona), ma la **validation continua a oscillare violentemente** (val_accuracy tra 0.70 e 0.93 tra epoche successive). Risultato: F ancora non appresa (F1 = 0.00), S quasi non appresa (F1 = 0.05), nonostante l'apparente stabilizzazione del training.

*Causa identificata*: la classe F ha solo 422 esempi su 75.395 nel training set (~0.36 esempi attesi per batch da 64) — la maggior parte dei batch non contiene alcun esempio di F, e i pochi che ne contengono causano aggiornamenti del gradiente sproporzionati per via del peso ancora elevato (10x), producendo il pattern di oscillazione osservato.

*Osservazione aggiuntiva*: `EarlyStopping` monitorava `val_loss`, una metrica dominata dalla classe maggioritaria N — non riflette la qualità reale su classi minoritarie e può interrompere il training in un punto non ottimale dal punto di vista clinico.

**Correzione adottata (terzo tentativo) — cambio di approccio strutturale:**
1. **Rimossi i class weights**, sostituiti da **campionamento stratificato dei batch** (`tf.data.Dataset.sample_from_datasets` con pesi fissi per classe: F 30%, N 20%, Q 15%, S 15%, V 20%) — garantisce presenza costante di ogni classe in ogni batch, eliminando la sparsità che causava gli aggiornamenti a scatto
2. **Introdotto un callback custom (`MacroF1Callback`)** che calcola il Macro F1 su validation a ogni epoca
3. **Early Stopping e ripristino dei pesi migliori spostati su `val_macro_f1`** (mode='max') invece che su `val_loss` — allinea la selezione del modello alla metrica clinicamente rilevante
4. `ReduceLROnPlateau` mantenuto su `val_loss` (solo per stabilità dell'ottimizzatore, non per selezione del modello)

**Bug tecnico incontrato e risolto durante l'implementazione**: `ValueError: target.shape=(None,), output.shape=(None, 5)` — il dataset stratificato costruito con `tf.data` produce etichette come interi singoli, non one-hot, mentre la loss `categorical_crossentropy` richiede il formato one-hot. Risolto passando a `sparse_categorical_crossentropy` (lavora nativamente con etichette intere) e usando `y_val_int` al posto di `y_val_cat` nella validation_data.

**Risultato Tentativo 3 (batch stratificato + monitoraggio Macro F1)**: miglioramento parziale ma non risolutivo. V ora eccellente (F1 0.90), ma **macro F1 sostanzialmente invariato (0.37 → 0.38)** e **F ancora a F1 = 0.00**, nonostante il 30% di presenza garantita in ogni batch. La loss di training scende fino a 0.08 (adattamento quasi perfetto ai dati visti), mentre la validation resta instabile con oscillazioni marcate dopo l'epoca 15.

**Diagnosi**: il campionamento stratificato risolve la *frequenza* con cui la rete vede esempi di F, ma non la *scarsità assoluta* — con soli 422 esempi distinti nel training set, ripresentati ripetutamente per 30 epoche, il modello mostra segni di **overfitting mirato sulla classe rara** (memorizza le particolarità dei pochi esempi specifici invece di generalizzare un pattern morfologico), risultando incapace di riconoscere esempi di F mai visti nel validation set. Conferma che il problema non era principalmente la composizione dei batch, ma la scarsità intrinseca di esempi diversi.

**Decisione presa**: non procedere con data augmentation sintetica (opzione B) per preferenza esplicita di restare fedeli a dati reali non perturbati artificialmente, nonostante la tecnica sia riconosciuta e validata in letteratura biomedica quando limitata a perturbazioni label-preserving (jitter temporale, rumore gaussiano lieve, scaling di ampiezza) su dati reali — diverso da generazione sintetica via modelli generativi (GAN), quella sì più controversa in letteratura.

**Prossimo tentativo (4) — Classificazione gerarchica a due stadi (opzione E)**: si procede con la ristrutturazione del problema in due classificatori in cascata:
- **Stadio 1**: classificatore binario Normale (N) vs Anomalo (S+V+F+Q raggruppati) — beneficia di molti più esempi aggregati nella classe minoritaria (~18.000 invece dei 422 isolati di F)
- **Stadio 2**: classificatore a 4 classi (S/V/F/Q) applicato solo ai casi già identificati come anomali dallo Stadio 1 — problema più piccolo e mirato, F non compete più contro l'enorme maggioranza N

**Risultato Tentativo 4 (pipeline a due stadi, valutazione end-to-end su validation)**: macro F1 = 0.37 — **sostanzialmente invariato** rispetto a tutti i tentativi precedenti (0.36 → 0.37 → 0.38 → 0.37). V leggermente peggiorato (F1 0.84, competeva contro Q nello Stadio 2). F e S restano a F1 ≈ 0.02, invariati rispetto a ogni tecnica di bilanciamento provata finora.

**Conclusione diagnostica**: avendo testato quattro approcci di bilanciamento sostanzialmente diversi (class weights semplici, class weights cappati, campionamento stratificato dei batch, ristrutturazione gerarchica del problema) senza alcun miglioramento sostanziale su F e S, si esclude empiricamente che la causa sia una tecnica di bilanciamento inadeguata. Le cause più probabili, non ancora affrontate, sono strutturali:
- **F**: 422 esempi totali nel training set sono probabilmente insufficienti in assoluto perché una CNN apprenda un pattern morfologico generalizzabile — nessuna tecnica di bilanciamento crea informazione dal nulla
- **S**: si distingue clinicamente più per il *timing* del battito (prematurità rispetto al ritmo atteso) che per la morfologia — informazione che il modello, vedendo solo la forma d'onda isolata, non riceve strutturalmente

**Decisione**: proseguire in modo più sperimentale, riconsiderando le opzioni precedentemente accantonate (Focal Loss, input ausiliario RR interval), accettando un approccio meno standard pur di ottenere un modello utilizzabile.

**Risultato Tentativo 5 (RR interval ausiliario + data augmentation su F, classificatore singolo a 5 classi)**: macro F1 = **0.41**, il migliore ottenuto finora (contro 0.36-0.38 dei tentativi precedenti).
- **S migliorata concretamente**: F1 da ≈0.02-0.05 a **0.16** — conferma la diagnosi che l'informazione di timing (RR interval) era il fattore mancante per questa classe
- **V migliorato**: F1 0.91
- **F rimasta a F1 = 0.00**, nonostante 3.376 esempi sintetici aggiunti tramite augmentation (jitter, rumore, scaling) — nessun miglioramento misurabile
- Validation ancora moderatamente instabile (oscillazioni tra epoche, seppur meno marcate dei tentativi 1-2)

**Nuova ipotesi su F (aggiornata)**: la classe F (fusione) potrebbe essere non solo rara ma **intrinsecamente ambigua** — per definizione clinica è un ibrido morfologico tra N e V, non una categoria con un pattern morfologico proprio e distinguibile. L'augmentation, generando varianti perturbate degli stessi 422 esempi originali, non introduce vera diversità morfologica: la rete continua a "risolvere" l'ambiguità collassando sistematicamente su N o V, le classi meglio definite e più popolate. Ipotesi da tenere in considerazione: **F potrebbe non essere risolvibile in modo soddisfacente con questo dataset/approccio**, per ragioni cliniche intrinseche più che tecniche.

**Decisione**: combinare le tecniche validate singolarmente (RR interval, monitoraggio Macro F1, gestione bilanciamento) in un unico Tentativo 6, invece di sostituire una tecnica con l'altra a ogni iterazione. Accettare la possibilità che F resti una limitazione nota e documentata del progetto, se il Tentativo 6 confermasse l'assenza di miglioramento.

**Risultato Tentativo 6 (RR interval + augmentation su F + class weights + monitoraggio Macro F1, classificatore singolo a 5 classi)**: macro F1 = **0.42**, ulteriore miglioramento rispetto al Tentativo 5 (0.41).
- **S migliorata sensibilmente**: F1 = **0.26** (recall 0.40) — il risultato migliore su questa classe finora, conferma che la combinazione di più tecniche (non solo RR da solo) aiuta S
- **V lievemente peggiorato**: F1 0.86 (da 0.91) — probabile compromesso dovuto al monitoraggio su Macro F1 invece che val_loss, che ora bilancia diversamente l'attenzione tra le classi
- **F ancora a F1 ≈ 0.00** — ulteriore conferma empirica (sesto tentativo consecutivo) che il problema di F non è risolvibile con tecniche di bilanciamento/training su un classificatore singolo, rafforzando l'ipotesi di ambiguità morfologica intrinseca
- Curva Macro F1 su validation: plateau rumoroso attorno a 0.40-0.42 dall'epoca 15 in poi, nessun miglioramento sostanziale nelle epoche successive

**Decisione finale**: combinare l'approccio gerarchico a due stadi (Tentativo 4) con tutte le tecniche mirate validate nei Tentativi 5-6 (RR interval, augmentation su F, monitoraggio Macro F1), applicando ogni tecnica allo stadio più pertinente:
- **Stadio 1** (N vs Anomalo): include RR interval come input ausiliario (il timing è rilevante anche per la separazione N/Anomalo, dato che S è tra le classi anomale)
- **Stadio 2** (S/V/F/Q su soli anomali): include RR interval, augmentation su F, class weights, monitoraggio Macro F1 — tutte le tecniche concentrate sul problema più piccolo e mirato, dove ci si aspetta il massimo beneficio cumulativo
- Valutazione end-to-end della pipeline combinata su validation, poi test una tantum

**Risultato Tentativo 7 — Stadio 1 (N vs Anomalo, con RR interval)**: precision Anomalo 0.97, **recall 0.74** (F1 0.84). Punto critico identificato: il 26% dei casi anomali reali (389 su 1490) viene classificato erroneamente come Normale a questo stadio — questi casi sono persi in modo definitivo per il resto della pipeline, ponendo un tetto massimo strutturale a quanto la pipeline nel suo complesso può recuperare, indipendentemente dalla qualità dello Stadio 2.

**Risultato Tentativo 7 — Stadio 2 (S/V/F/Q su soli anomali, con RR + augmentation + Macro F1)**: risultato più significativo del progetto finora — **F1 F = 0.39** (precision 0.34, recall 0.45), primo miglioramento sostanziale su questa classe dopo sei tentativi consecutivi a F1 ≈ 0.00. V eccellente (F1 0.80), S ancora debole (F1 0.18), Q assente nel fold di validation (0 esempi, coerente con l'osservazione già nota). Macro F1 sulle 3 classi effettivamente presenti ≈ 0.46 (picco all'epoca 2, poi lieve calo e plateau — pesi ripristinati correttamente dall'Early Stopping su val_macro_f1).

**Conferma diagnostica importante**: il miglioramento di F conferma che il problema non era la classe in sé irrecuperabile, ma la sua schiacciante minoranza nel problema a 5 classi originale (F contro l'enorme maggioranza N). Isolandola in un sottoproblema a 4 classi (dove compete solo con S/Q/V), la combinazione di augmentation e RR interval è riuscita a farle apprendere un pattern morfologico reale.

**Bug tecnico incontrato**: `classification_report` di sklearn solleva `ValueError` quando il numero di classi effettivamente presenti nei dati (es. 3, per assenza di Q nel fold di validation) non coincide con la lunghezza di `target_names` (5, tutte le classi attese). Risolto passando esplicitamente il parametro `labels=list(range(num_classi))` sia a `confusion_matrix` che a `classification_report`, forzando la valutazione a considerare sempre tutte le classi attese anche quando alcune non sono rappresentate nei dati.

**Nota sulla valutazione finale della pipeline combinata**: attesa una performance intermedia tra Stadio 2 isolato e il limite imposto dal recall dello Stadio 1 (74% sugli anomali) — il risultato finale end-to-end sconterà inevitabilmente i casi anomali persi al primo stadio.

**Risultato Tentativo 7 — Pipeline combinata finale (validation)**: macro F1 = 0.37, **inferiore** al Tentativo 6 (0.42) e nettamente inferiore alla performance dello Stadio 2 isolato su F (0.39 → crollato a **0.01** nella pipeline finale). Causa identificata dalla confusion matrix: 273 casi reali di F su 366 vengono classificati come N già allo Stadio 1, prima di raggiungere lo Stadio 2 che li avrebbe riconosciuti correttamente — il collo di bottiglia dello Stadio 1 colpisce F in modo sproporzionato rispetto alle altre classi anomale, coerente con l'osservazione clinica che la fusione è morfologicamente "quasi normale" per definizione.

**Conclusione**: il classificatore singolo del Tentativo 6 (macro F1 0.42, architettura unica con RR interval + augmentation + class weights + monitoraggio Macro F1) resta il modello più efficace end-to-end. L'approccio gerarchico ha prodotto un'informazione scientificamente utile (F si separa bene una volta isolata dalla maggioranza N) ma introduce un secondo punto di fallimento (recall imperfetto dello Stadio 1) che nei fatti annulla il guadagno. **Il Tentativo 6 è il modello di riferimento per gli sviluppi successivi**, non la pipeline gerarchica.

**Osservazione aggiuntiva**: la balanced accuracy (media dei recall per classe) è stata considerata come metrica alternativa/complementare al macro F1, ma non risolve il problema della classe Q assente nel fold di validation (stesso problema di indefinizione matematica per classi a zero esempi). Verrà comunque affiancata al macro F1 nei report successivi per maggiore interpretabilità.

**Decisione — passaggio a GroupKFold Cross-Validation**: per ottenere una stima onesta anche sulla classe Q (mai presente nel singolo fold di validation fisso a causa della sua concentrazione in pochi record specifici) e una valutazione più robusta sulle classi rare in generale, si adotta **GroupKFold** (k=5, raggruppato per record/paziente) applicato al modello di riferimento (Tentativo 6: RR interval + augmentation su F + class weights + monitoraggio Macro F1). Ogni fold usa split diversi di record per training/validation, garantendo che nel complesso di tutti i fold ogni classe (inclusa Q) venga valutata almeno una volta. Metriche aggregate (media e deviazione standard del macro F1 e F1 per classe tra i fold) sostituiranno la singola valutazione su fold fisso come stima di riferimento del progetto.

**Implementazione**: 5 training completi indipendenti, ciascuno con normalizzazione RR interval e augmentation su F ricalcolate esclusivamente sul training set di quel fold (per evitare leakage tra fold). Aggiunto meccanismo di checkpoint su Google Drive (salvataggio incrementale dopo ogni fold) per resistere a disconnessioni della sessione Colab durante l'esecuzione (~90-150 sec/epoca, 15-32 epoche/fold, diverse ore totali). Aggiunta successivamente la registrazione della confusion matrix per fold (non presente nella prima esecuzione), richiedendo una seconda esecuzione completa del k-fold.

**RISULTATO DEFINITIVO — GroupKFold Cross-Validation (5 fold, con confusion matrix)**, adottato come stima di riferimento ufficiale del progetto:

**Risultati per fold**:

| Fold | Record in validation (n=9-10) | Macro F1 | Balanced Acc |
|---|---|---|---|
| 1 | 100,102,108,109,118,119,201,210,215 | 0.5766 | 0.5901 |
| 2 | 105,106,111,115,116,124,205,213,230,231 | 0.4080 | 0.4635 |
| 3 | 103,107,117,121,122,200,212,217,232,233 | 0.5725 | 0.5702 |
| 4 | 101,207,209,214,219,222,223,228,234 | 0.4099 | 0.4079 |
| 5 | 104,112,113,114,123,202,203,208,220,221 | 0.6198 | 0.7335 |

| Metrica | Media | Deviazione standard |
|---|---|---|
| Macro F1 | **0.5174** | 0.0901 |
| Balanced Accuracy | 0.5530 | 0.1125 |

**F1 medio per classe** (media semplice sui 5 fold, ciascuno pesato ugualmente indipendentemente dal numero di esempi):

| Classe | F1 medio |
|---|---|
| N | 0.8930 |
| V | 0.8565 |
| Q | 0.5516 |
| S | 0.2617 |
| F | 0.0241 |

**Nota su variabilità run-to-run**: questa seconda esecuzione del GroupKFold (stessi split, dato che GroupKFold è deterministico sugli stessi dati) produce macro F1 medio 0.52 contro lo 0.48 della prima esecuzione — differenza attribuibile alla stocasticità intrinseca del training (inizializzazione pesi, ordine dei batch), non alla composizione dei fold. Ulteriore fonte di variabilità da tenere presente oltre a quella tra fold: **anche a parità di split, run diverse dello stesso identico esperimento producono risultati non identici**.

**Confusion Matrix Aggregata (somma dei 5 fold) — Analisi**:

Normalizzata per riga (percentuale di ogni classe reale finita in ciascuna predizione):

| Reale \ Predetto | F | N | Q | S | V |
|---|---|---|---|---|---|
| F | 0.46 | 0.44 | 0.00 | 0.01 | 0.09 |
| N | 0.06 | 0.85 | 0.07 | 0.02 | 0.01 |
| Q | 0.03 | 0.10 | 0.87 | 0.00 | 0.00 |
| S | 0.03 | 0.62 | 0.00 | 0.28 | 0.07 |
| V | 0.04 | 0.04 | 0.01 | 0.02 | 0.88 |

**Osservazioni chiave dalla confusion matrix aggregata**:
- **F si confonde quasi esclusivamente con N (44%) e in misura minore con V (9%)** — mai con Q o S. Conferma quantitativa, per la prima volta con dati aggregati su tutto il dataset, dell'ipotesi di ambiguità morfologica specifica tra F/N/V (non un'ambiguità generica) formulata dopo i Tentativi 5-7. La ripartizione 44% N / 9% V (non fortemente sbilanciata verso una sola delle due) suggerisce che F non è chiaramente più vicina a N che a V, o viceversa — è genuinamente intermedia
- **Q ha recall aggregato ≈87%** (buono) ma **precision moderata**: 6.375 esempi di N vengono erroneamente classificati come Q (falsi positivi consistenti da N verso Q)
- **S ha recall aggregato solo 28%**, con il 62% degli esempi reali di S classificati come N — conferma che il problema del timing/prematurità resta solo parzialmente catturato nonostante l'RR interval
- **V solidamente riconosciuta**: recall 88%, confusione minima e distribuita

**Discrepanza tra F1 medio per-fold (0.024 per F) e recall aggregato (46% per F)**: spiegata dal fatto che il Fold 5 contiene una quota sproporzionata di esempi F (378 su 802 totali) con un comportamento anomalo del modello in quel fold (recall 97% ma precision 6% — il modello ha praticamente etichettato quasi tutto come F pur di non mancarla, sacrificando N, la cui recall in quel fold crolla al 73%). La media semplice per-fold pesa ugualmente ciascun fold, quindi non riflette questo comportamento estremo quanto la matrice aggregata. Entrambe le viste sono corrette ma rispondono a domande diverse: il valore per-fold è più rappresentativo della "tipica" performance su un paziente casuale, il valore aggregato è più rappresentativo della performance su tutto il dataset combinato.

### 4.15 — Decisione sulla gestione della classe F: soglia di confidenza (reject option)

Valutate tre opzioni per la gestione della classe F, alla luce dei dati quantitativi emersi dalla confusion matrix aggregata:
- **A — Escludere F** dal problema (4 classi): semplifica ma nasconde il limite, un caso reale di fusione verrebbe sempre forzato in un'altra categoria senza segnalazione
- **B — Soglia di confidenza (reject option)**: se la probabilità massima del softmax è sotto una soglia, il sistema restituisce "incerto" invece di forzare una classe, segnalando il caso per revisione umana
- **C — Accorpare F a N o V**: la confusion matrix aggregata (44% F→N, 9% F→V) non mostra una prevalenza netta verso una delle due, rendendo la scelta arbitraria

**Decisione: opzione B (soglia di confidenza)**, motivata dallo standard consolidato in ambito biomedicale/clinico di ML — noto in letteratura come *selective prediction* o *reject option classifier*. Principio guida: in un contesto diagnostico, un **falso negativo silenzioso rappresenta il rischio maggiore**; un sistema che comunica esplicitamente la propria incertezza consente l'intervento umano, mentre uno che forza sempre una risposta nasconde il rischio. Rafforzata dall'osservazione empirica (Fold 5) che il modello, quando "spinto" verso un equilibrio diverso, dimostra di possedere segnale utile su F (recall 97%) — il problema non è l'assenza totale di informazione, ma la scelta di come e quando esprimerla con sicurezza.

**Implementazione**: funzione `predici_con_soglia()` che classifica un segmento solo se la confidenza massima del softmax supera una soglia impostabile, altrimenti restituisce `'INCERTO'`. La soglia ottimale verrà scelta valutando il trade-off tra coverage (percentuale di casi classificati) e macro F1 sui soli casi confidenti, oltre alla percentuale di casi reali di F correttamente "salvati" dalla soglia (marcati incerti invece che silenziosamente sbagliati) — analisi da eseguire sul modello finale.

### 6.5 — Decisione: continuare l'ottimizzazione invece di procedere al deploy

Nonostante il risultato delle sezioni 6.2-6.4 sia documentabile e scientificamente onesto, si decide di non fermarsi qui: l'obiettivo del progetto richiede un prototipo con performance più solide, non solo un percorso di sviluppo ben documentato. Identificate due leve concrete non ancora esplorate, con potenziale di impatto alto e costo di implementazione contenuto:

**Leva 1 — Soglia di decisione dello Stadio 1 (pipeline gerarchica)**: finora sempre usata la soglia di default (0.5) per il classificatore binario N vs Anomalo. Il recall misurato sugli anomali era solo 74% (Tentativo 7) — il vero collo di bottiglia identificato, mai affrontato direttamente. Abbassando la soglia di decisione (es. a 0.3), si favorisce il recall a scapito della precisione su N, recuperando una quota maggiore di casi anomali (incluso F) da inoltrare allo Stadio 2 — dove sappiamo che F raggiunge F1=0.39 quando correttamente isolata dalla maggioranza N.

**Leva 2 — Calibrazione della soglia di confidenza per classe**: la scoperta che Q viene classificato con sicurezza (ma erroneamente) come N nel 79% dei casi reali suggerisce che una soglia di confidenza globale uguale per tutte le classi non è ottimale. Da valutare una soglia specifica per classe, basata sulla distribuzione di confidenza osservata per ciascuna sul validation.

**Piano**: ripresa della pipeline gerarchica a due stadi (Tentativo 7), con soglia di decisione dello Stadio 1 ottimizzata per il recall sugli anomali, e soglia di confidenza per classe (non globale) applicata allo Stadio 2. Nuova valutazione end-to-end attesa.

### 4.16 — Consolidamento del notebook in un unico file

Unificati in un solo notebook autosufficiente (`ecg_cnn_progetto_completo.ipynb`) tutti i passaggi precedentemente distribuiti su file separati: preparazione dati, GroupKFold Cross-Validation (con confusion matrix per fold, correzione rispetto alla prima esecuzione), e pipeline gerarchica finale con soglie ottimizzate. Corretto contestualmente un problema di persistenza: il meccanismo di checkpoint/ripresa del GroupKFold (salvataggio incrementale su Drive dopo ogni fold) era andato perso nella prima fusione dei notebook — reintrodotto con file di checkpoint rinominato (`groupkfold_risultati_v2.pkl`) per includere le confusion matrix mancanti nella versione precedente.

**Pulizia Google Drive**: rimossi i file intermedi non più necessari con la struttura dati attuale (split fissi `X_train.npy`/`X_val.npy`/`X_test.npy` e derivati, sostituiti dal ricalcolo dello split al volo da `X_full.npy`/`y_int_full.npy`/`record_id_full.npy`/`rr_full.npy`), il vecchio checkpoint GroupKFold senza confusion matrix, e il modello finale del classificatore singolo a 5 classi (superato dall'approccio gerarchico).

**Riconferma GroupKFold (seconda esecuzione, con confusion matrix)**: macro F1 media = 0.5005 (std 0.0688), coerente con la prima esecuzione (0.4807-0.5174) — F1 per classe: N 0.9158, V 0.7648, Q 0.5542, S 0.2155, F 0.0524. Risultati stabili e riproducibili nell'ordine di grandezza, confermando l'affidabilità della stima di riferimento.

### 4.17 — Tentativo 8: pipeline gerarchica con soglie ottimizzate — RISULTATO NEGATIVO, non adottato

**Leva 1 (soglia Stadio 1 per il recall)**: risultato genuinamente positivo. Soglia ottimale trovata a 0.30 (invece del default 0.5): recall sugli anomali migliorato da 0.74 (Tentativo 7) a **0.81**, con precision 0.69 — miglioramento reale e verificabile del collo di bottiglia identificato in precedenza.

**Leva 2 (calibrazione soglia di confidenza per classe, Stadio 2)**: **fallimento metodologico riconosciuto**. Le soglie trovate per S (F1=1.0000) e Q (F1=0.9993) sul validation interno erano sospette fin da subito — F1 pressoché perfetti su classi notoriamente difficili sono quasi sempre sintomo di overfitting alla ricerca della soglia stessa su un campione piccolo (il validation anomalo conta solo poche decine/centinaia di esempi per classe), non un vero miglioramento generalizzabile.

**Conferma del fallimento nei risultati**:
- Validation interno: macro F1 = 0.43 (già inferiore al Tentativo 6/7)
- **Test set: macro F1 = 0.3558** — il risultato peggiore di tutto il progetto. Q crolla a F1=0.00 (nonostante la soglia "quasi perfetta" trovata sul validation), S crolla a F1=0.11
- Il miglioramento genuino della Leva 1 (soglia Stadio 1) è stato interamente vanificato dal danno introdotto dalla Leva 2 a valle

**Violazione metodologica riconosciuta esplicitamente**: questo costituisce il **secondo utilizzo del test set** nel progetto (il primo: Tentativo 6/7, macro F1 0.4488). La regola "test set toccato una sola volta" concordata a inizio progetto è stata infranta nel tentativo di ottenere un risultato migliore. Si dichiara questo scostamento esplicitamente invece di ometterlo, insieme alla lezione appresa: la calibrazione di soglie per classe su campioni ridotti richiede cautela metodologica (es. validazione nidificata, campioni minimi garantiti per classe) che non era stata applicata qui.

**Decisione finale**: il Tentativo 8 non viene adottato come risultato ufficiale del progetto. **Si conferma il Tentativo 6/7 (classificatore singolo a 5 classi, soglia di confidenza globale, macro F1 sul test = 0.4488) come risultato di riferimento definitivo**. Il Tentativo 8 viene documentato come esperimento negativo istruttivo: dimostra concretamente il rischio di overfitting nella calibrazione di iperparametri di post-processing su campioni piccoli, un contenuto di valore metodologico per il portfolio a sé stante, pur non avendo migliorato il modello.

---

## 6. RISULTATO FINALE DEL PROGETTO



### 6.1 — Modello finale e scelta della soglia

Modello riallenato su train+validation uniti (85% dei record, split interno 15% riservato al solo Early Stopping), stessa ricetta consolidata (RR interval + augmentation su F + class weights + monitoraggio Macro F1).

**Analisi soglia di confidenza sul validation interno**:

| Soglia | Coverage | Macro F1 (sui confidenti) | F segnalati incerti |
|---|---|---|---|
| 0.3 | 100.0% | 0.5234 | 0/2 |
| 0.4 | 99.9% | 0.5246 | 0/2 |
| 0.5 | 99.4% | 0.5270 | 0/2 |
| **0.6** | **97.8%** | **0.5406** | 1/2 |
| 0.7 | 94.8% | 0.5410 | 1/2 |
| 0.8 | 87.7% | 0.5552 | 1/2 |
| 0.9 | 61.5% | 0.4709 | 2/2 |

**Soglia scelta: 0.6** — buon compromesso tra coverage alta (97.8%) e miglioramento della qualità sui casi confidenti, senza scartare una quota eccessiva di dati (a differenza di 0.9, che scarta quasi il 40% del dataset). *Nota: il validation interno conteneva solo 2 esempi di F, campione troppo piccolo per guidare da solo la scelta della soglia specificamente su quella classe.*

### 6.2 — Valutazione conclusiva sul Test Set (unico utilizzo, soglia=0.6)

**Coverage sul test**: 87.6% dei casi classificati con confidenza sufficiente, 12.4% segnalati `INCERTO`.

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| F | 0.00 | 0.00 | 0.00 | 13 |
| N | 0.91 | 0.88 | 0.89 | 14.147 |
| Q | 0.99 | 0.20 | 0.34 | 1.428 |
| S | 0.16 | 0.79 | 0.27 | 229 |
| V | 0.60 | 0.98 | 0.74 | 955 |

**MACRO F1 FINALE SUL TEST SET: 0.4488**

### 6.3 — Analisi onesta dei risultati finali

**Scoperta importante sul limite della soglia di confidenza**: la confusion matrix rivela che **1.133 casi reali di Q su 1.428 (79%) vengono classificati con sicurezza (confidenza ≥0.6) come N**, non segnalati come incerti. Questo dimostra un limite intrinseco della tecnica reject-option: **funziona solo quando il modello stesso "sa" di essere incerto** (bassa confidenza softmax); se invece è confidente ma sbagliato (overconfidence), la soglia non lo intercetta. Risultato scientificamente onesto da riportare, non un fallimento dell'approccio ma un suo limite noto e documentato in letteratura sul reject-option.

**Pattern precision/recall asimmetrico su S e V**: entrambe mostrano recall alto (S: 0.79, V: 0.98) ma precision bassa (S: 0.16, V: 0.60) — il modello tende a etichettare troppi casi N come S o V (falsi positivi), generando più falsi allarmi che falsi negativi silenziosi. Dal punto di vista della sicurezza clinica, questo è il tipo di errore preferibile (un falso allarme richiede solo verifica aggiuntiva, un falso negativo rischia di passare inosservato), ma va comunque dichiarato come limite di specificità del dispositivo.

**F**: 0/13 corretti, ma il campione (13 esempi) è troppo piccolo per essere statisticamente significativo — consistente con quanto già osservato nel GroupKFold, dove F risultava recuperabile solo in condizioni specifiche di training (Fold 5) e restava comunque la classe più fragile del progetto.

**Confronto con la stima GroupKFold**: macro F1 = 0.45 sul test, contro una media di 0.52 (range 0.41-0.62) nei 5 fold di cross-validation — il risultato è nella parte bassa ma **dentro il range di variabilità già documentato**, non un segnale di overfitting o di errore metodologico, semplicemente l'esito di questo specifico training/split.

### 6.4 — Limiti noti e dichiarati del progetto

1. **Classe F**: non recuperabile in modo affidabile con le tecniche esplorate (RR interval, augmentation, class weights, approccio gerarchico, soglia di confidenza) — ambiguità morfologica intrinseca con N e V, confermata quantitativamente dalla confusion matrix aggregata (44% F→N, 9% F→V)
2. **Classe Q**: buona precision ma recall variabile e talvolta basso — il modello può classificare con sicurezza (ma erroneamente) casi di Q come N, limite non mitigato dalla soglia di confidenza
3. **Classi S e V**: buon recall ma precision limitata — tendenza a falsi positivi (over-triggering) più che a falsi negativi
4. **Variabilità run-to-run e tra split**: risultati non pienamente riproducibili a parità di dati, per la stocasticità intrinseca del training di reti neurali — mitigata parzialmente da Early Stopping su Macro F1, ma non eliminata
5. **Domain gap atteso**: tutte le valutazioni sono su MIT-BIH (segnale clinico controllato); performance su segnale reale da MAX30003 con elettrodi posizionati manualmente sarà verosimilmente inferiore, da validare empiricamente in fase di test del dispositivo fisico

### 4.13 — Persistenza dati tra sessioni Colab
Problema riscontrato: disconnessioni ripetute del runtime Colab (per inattività o scadenza sessione) causano la perdita di tutte le variabili in memoria, costringendo a rieseguire l'intera pipeline di download ed elaborazione (Sezioni 1-6, ~8-10 minuti) ogni volta.

**Soluzione adottata**: salvataggio di `X_train/y_train/X_val/y_val/X_test/y_test` (dataset già segmentato, filtrato, normalizzato e diviso per record) su Google Drive in formato `.npy`, tramite `google.colab.drive`.

**Percorso di salvataggio**: `/content/drive/MyDrive/ecg_project_data/` — cartella dedicata creata automaticamente dal codice (`os.makedirs(..., exist_ok=True)`) nella root del Drive, separata dalla cartella "Colab Notebooks" dove risiede il file `.ipynb`. Contiene i 6 file `.npy` (train/val/test per X e y).

**Riorganizzazione del notebook**: aggiunta una **Sezione 0 (Avvio rapido)** in cima al notebook, prima della Sezione 1, e una **Sezione 6bis (Salvataggio su Drive)** subito dopo la Sezione 6 (split). Flusso d'uso:
- **Prima esecuzione / rigenerazione completa**: Sezioni 1→2→3→4→5→6→6bis→7→8→9→10→11 in ordine
- **Sessioni successive** (dopo disconnessione): solo Sezione 0 (mount Drive + caricamento `.npy`) seguita direttamente da 7→8→9→10→11, saltando le Sezioni 1-6bis (che restano nel notebook come documentazione della costruzione del dataset, non vengono rieseguite)

L'autorizzazione di accesso a Google Drive (popup di consenso account Google) va ripetuta a ogni nuova sessione, non è persistente tra riavvii.

### 4.14 — Nota metodologica: validation set e selezione ripetuta del modello
Il validation set è stato osservato e usato per decisioni ripetutamente attraverso i Tentativi 1-4 (class weights, capping, batch stratificato, approccio a due stadi) — un fenomeno noto in letteratura come *leakage da selezione ripetuta del modello* ("researcher degrees of freedom"), distinto dal data leakage strutturale (già escluso dallo split per paziente). Ogni iterazione che tiene conto del risultato di validation per decidere il passo successivo riduce leggermente la purezza di quel set come stima di generalizzazione.

**Mitigazione**: il test set non è mai stato osservato né usato per alcuna decisione nei Tentativi 1-4. Verrà valutato una sola volta, a fine iterazione, quando la pipeline a due stadi risulterà soddisfacente su validation — quel numero costituirà la stima finale onesta di generalizzazione riportata nel progetto.

## 5. Dataset

### 5.1 — MIT-BIH Arrhythmia Database
- **Fonte**: PhysioNet, physionet.org/content/mitdb/1.0.0/
- **Accesso**: libreria Python `wfdb`, download diretto dei record via `pn_dir='mitdb'` (no download manuale necessario)
- **Struttura**: 48 record, 2 canali ECG per record, frequenza di campionamento **360 Hz**, durata 30 minuti ciascuno
- **Annotazioni**: ogni battito ha una posizione (in campioni) e un simbolo che ne indica il tipo

### 5.2 — Nomenclatura record
I 48 record sono numerati in due serie non contigue per motivi storici:
- **100-124**: casistica "di routine" (mix rappresentativo di aritmie comuni)
- **200-234**: casistica arricchita di aritmie rare/complesse (blocchi di branca, fusioni, episodi di fibrillazione ventricolare)

Alcuni numeri mancano in entrambi i range (record scartati/non inclusi nella versione pubblicata finale).

### 5.3 — Simboli di annotazione: battito vs marcatore tecnico
Non tutti i simboli di annotazione rappresentano un battito classificabile. Esclusi dal dataset di training perché sono metadati tecnici, non classi cliniche:

| Simbolo | Significato |
|---|---|
| `+` | Cambio di ritmo (rhythm change marker) |
| `~` | Cambio qualità segnale (noise marker) |
| `"` | Commento annotazione |
| `\|` | Battito isolato/qualità dubbia |
| `[` `]` | Inizio/fine episodio di fibrillazione ventricolare |
| `x` | Marcatore ausiliario legato a episodi VF |

### 5.4 — Mappatura simboli MIT-BIH → classi AAMI

| Classe AAMI | Simboli MIT-BIH inclusi | Significato clinico |
|---|---|---|
| **N** (Normal) | N, L, R, e, j | Battito normale o variante normale |
| **S** (Supraventricular) | A, a, J, S | Origine sopraventricolare |
| **V** (Ventricular) | V, E | Origine ventricolare |
| **F** (Fusion) | F | Fusione normale/ventricolare |
| **Q** (Unknown/paced) | /, f, Q | Non classificabile o paced |

### 5.5 — Distribuzione classi sull'intero dataset (48 record)

Distribuzione grezza per simbolo (prima del raggruppamento AAMI), su un totale di ~112.000 annotazioni:

| Simbolo | Conteggio | | Simbolo | Conteggio |
|---|---|---|---|---|
| N | 75.052 | | f | 982 |
| L | 8.075 | | F | 803 |
| R | 7.259 | | ~ | 616 |
| V | 7.130 | | ! | 472 |
| / | 7.028 | | " | 437 |
| A | 2.546 | | j | 229 |
| + | 1.291 | | x | 193 |

**Distribuzione finale dopo raggruppamento AAMI** (confermata su tutti i 48 record):

| Classe AAMI | Conteggio | % sul totale |
|---|---|---|
| N (Normal) | 90.631 | ~82.7% |
| Q (Unknown/paced) | 8.043 | ~7.3% |
| V (Ventricular) | 7.236 | ~6.6% |
| S (Supraventricular) | 2.781 | ~2.5% |
| F (Fusion) | 803 | ~0.7% |

Totale battiti utilizzabili: 109.494.

**Osservazione chiave**: forte sbilanciamento di classe (N ≈ 83% del totale, F meno dell'1%). Rapporto tra classe più numerosa e meno numerosa: ~113:1. Da gestire in fase di training con tecniche dedicate (class weights, oversampling — non ancora implementate, pianificate per il Blocco 3).

---

## 6. Software e toolchain

**Struttura del notebook Colab**: organizzato in sezioni numerate, ciascuna con una cella Markdown descrittiva (cosa/perché) seguita dalle celle di codice corrispondenti.

| Sezione | Contenuto | Da rieseguire ogni sessione? |
|---|---|---|
| 0 | Avvio rapido — carica dataset processato da Drive | Sì (alternativa a 1-6bis) |
| 1 | Setup ambiente | Solo se si riparte da zero |
| 2 | Esplorazione preliminare (record singolo) | Solo se si riparte da zero |
| 3 | Download completo dataset + mappatura AAMI | Solo se si riparte da zero |
| 4 | Segmentazione dei battiti | Solo se si riparte da zero |
| 5 | Preprocessing (filtro + normalizzazione) | Solo se si riparte da zero |
| 6 | Split Train/Validation/Test per record | Solo se si riparte da zero |
| 6bis | Salvataggio dataset processato su Google Drive | Solo dopo aver rigenerato da zero |
| 7 | Gestione sbilanciamento classi (class weights + capping) | Sì |
| 8 | Codifica etichette | Sì |
| 9 | Architettura CNN 1D | Sì |
| 10 | Compilazione e training | Sì |
| 11 | Valutazione (curve, confusion matrix, classification report) | Sì |

| Fase | Strumento | Ambiente |
|---|---|---|
| Esplorazione dati | Python + `wfdb` + `matplotlib` | Google Colab |
| Training rete neurale | TensorFlow/Keras | Google Colab |
| Conversione modello | TensorFlow Lite Converter (quantizzazione INT8) | Google Colab |
| Generazione codice C embedded | STM32CubeIDE + plugin X-CUBE-AI | Locale |
| Firmware/debug MCU | STM32CubeIDE (HAL STM32) | Locale |
| Libreria di riferimento sensore | ProtoCentral MAX30003 ECG AFE Sensor Library (Arduino, da adattare a HAL) | Locale |

---

## 7. Log delle attività

### Blocco 1 — Esplorazione dataset MIT-BIH
- Setup ambiente Colab, installazione `wfdb`
- Download ed esplorazione struttura del record 100 (fs=360Hz, 2 canali, 30 min)
- Visualizzazione tracciato con annotazioni sovrapposte
- Identificato outlier/artefatto nel record 100 (ampiezza anomala ~-2.5mV) — verosimile artefatto da movimento/elettrodo, non patologia
- Conteggio simboli di annotazione sul singolo record 100 → forte sbilanciamento locale (2239 N vs 33 A vs 1 V)
- Esteso il conteggio a tutti i 48 record → distribuzione completa ottenuta (vedi sezione 5.5)
- Definita mappatura simboli MIT-BIH → classi AAMI (5 macro-classi) per gestire simboli rari e allinearsi allo standard di letteratura
- Scritto codice per il download completo di segnale + annotazioni di tutti i 48 record, con filtraggio dei marcatori tecnici e applicazione della mappatura AAMI
- Eseguito il download completo: 109.494 battiti utilizzabili totali su tutti i 48 record
- Confermata distribuzione finale per classe AAMI: N 90.631 (82.7%) — Q 8.043 (7.3%) — V 7.236 (6.6%) — S 2.781 (2.5%) — F 803 (0.7%)
- **Blocco 1 completato**

### Blocco 2 — Segmentazione dei battiti
- Implementata segmentazione: finestra 300 campioni (100 prima + 200 dopo il picco R), canale 0, per tutti i battiti validi
- Estratti 109.446 segmenti su 109.494 battiti totali (48 scartati per vicinanza ai bordi del record)
- Verifica visiva: confronto qualitativo tra un segmento classe N e uno classe V — morfologia coerente con la letteratura clinica (QRS stretto e netto per N; QRS largo/bizzarro con onda T discordante per V/PVC), buon segnale che la segmentazione cattura correttamente le differenze diagnostiche rilevanti
- Applicato filtro passa-banda Butterworth (0.5-40Hz) sull'intero segnale di ogni record, prima della segmentazione
- Ri-segmentazione con normalizzazione z-score per finestra applicata dopo il filtraggio
- Risolto bug di indicizzazione nella verifica visiva (wraparound Python su indice negativo) tracciando esplicitamente la posizione originale di ogni battito
- Implementato split Train/Validation/Test per record (70/15/15): 33 record training (75.395 segmenti), 7 validation (14.911), 8 test (19.140)
- Verificata coerenza della distribuzione di classe tra training set e dataset completo (nessuno squilibrio aggiuntivo introdotto dallo split)
- **Blocco 2 completato**

### Blocco 3 — Sbilanciamento classi e architettura CNN (iterazioni)
- Calcolati class weights bilanciati sul training set (F: 35.7, S: 6.5, Q: 2.5, V: 2.9, N: 0.25)
- Codifica etichette da stringa a intero e poi one-hot, mappa: F=0, N=1, Q=2, S=3, V=4
- Definita architettura CNN 1D a 3 blocchi convoluzionali con GlobalAveragePooling1D (14.853 parametri, 58.02 KB)
- **Tentativo 1**: training con class weights non cappati (F=35.7x) — instabilità severa, macro F1 = 0.36
- **Tentativo 2**: class weights cappati a 10x + learning rate ridotto (0.0003) + ReduceLROnPlateau — training di training pulito, ma validation ancora instabile, macro F1 = 0.37, F1 F = 0.00, F1 S = 0.05
- **Tentativo 3**: abbandonati i class weights, adottato campionamento stratificato dei batch + monitoraggio Macro F1 per Early Stopping — risolto bug tecnico (sparse_categorical_crossentropy al posto di categorical_crossentropy). Risultato: V migliorato (F1 0.90), macro F1 sostanzialmente invariato (0.38), F ancora a 0 — diagnosticato overfitting mirato sulla classe rara per scarsità assoluta di esempi distinti (422), non per composizione dei batch
- Valutata e scartata l'opzione data augmentation (jitter/rumore/scaling) per scelta di restare su dati reali non perturbati, pur riconoscendone la validità scientifica in letteratura biomedica se applicata come perturbazione label-preserving
- **Tentativo 4 (deciso, da implementare)**: passaggio a classificazione gerarchica a due stadi — Stadio 1 binario N vs Anomalo, Stadio 2 a 4 classi (S/V/F/Q) solo sui casi anomali

- **Tentativo 4**: implementata pipeline gerarchica a due stadi (Stadio 1 binario N/Anomalo, Stadio 2 su 4 classi S/V/F/Q solo su anomali). Risultato end-to-end su validation: macro F1 = 0.37, invariato rispetto ai tentativi precedenti; F e S restano non apprese
- **Conclusione empirica**: quattro tecniche di bilanciamento diverse testate senza risultato su F/S — causa esclusa essere la tecnica di bilanciamento, spostata verso limiti strutturali (scarsità assoluta per F, mancanza di informazione di timing per S)
- **Decisione**: approccio più sperimentale, riconsiderando Focal Loss e input ausiliario RR interval, precedentemente accantonati

**Opzioni alternative identificate e non ancora testate** (se il Tentativo 4 fosse insufficiente):
- Focal Loss al posto della loss pesata/campionamento stratificato
- Input ausiliario: intervallo RR come feature numerica in parallelo al ramo convoluzionale (utile in particolare per la classe S, che si distingue più per timing che per morfologia)
- Valutata e scartata per ora l'estensione ad altri dataset: PTB-XL richiederebbe un cambio di paradigma (annotazioni per registrazione, non per battito); Icentia11k resta un'opzione compatibile con la pipeline attuale se il problema fosse di scarsità di dati piuttosto che di limite strutturale del modello

### Prossimi passi pianificati
- **Tentativo 6 (specifica pronta, da implementare)**: combinare in un'unica pipeline le tecniche validate singolarmente
  1. RR interval come input ausiliario (confermato utile per S — mantenere)
  2. Monitoraggio Macro F1 per Early Stopping/ripristino pesi (non più val_loss dominato da N)
  3. Data augmentation su F (mantenere nonostante il risultato nullo del Tentativo 5, per completezza sperimentale — non peggiora)
  4. Class weights bilanciati senza capping estremo (l'augmentation riduce già parzialmente lo sbilanciamento assoluto)
  5. Valutare anche batch stratificato in combinazione, se il tempo lo permette
- Se il Tentativo 6 confermasse F non recuperabile: documentare come limite noto del progetto (ambiguità clinica intrinseca della classe fusione + scarsità), procedere verso quantizzazione/deploy accettando questa limitazione
- Solo dopo aver ottenuto un modello soddisfacente su validation: valutazione finale una tantum su test set
- Conversione e quantizzazione INT8, verifica flash/RAM con X-CUBE-AI

---

## 8. Limiti noti e punti da validare in seguito

- **Domain gap** atteso tra segnale MIT-BIH (registrazione clinica controllata) e segnale reale acquisito con MAX30003 + elettrodi posizionati manualmente (rumore, artefatti da movimento diversi)
- Necessità di validare la stima di flash/RAM del modello finale con il tool di analisi X-CUBE-AI prima del deploy, per confermare la compatibilità con lo STM32F411
- Sbilanciamento di classe da gestire esplicitamente in fase di training, non ancora implementato
