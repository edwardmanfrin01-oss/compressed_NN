# Helper ML

- `helper_functions.py`: `train_step`, `test_step`, `summarize_training_time`.
  `train_step` restituisce loss, accuracy percentuale e training_seconds.
  Il tempo misura il ciclo dell'epoca (dati, trasferimenti, forward/backward,
  aggiornamenti); su CUDA sincronizza prima e dopo la misura.
  `test_step` restituisce loss, accuracy percentuale e confusion_counts come
  lista di liste (righe vere, colonne predette). Il numero di classi e un argomento.
  Usare loss scalare media per campione, come CrossEntropyLoss non pesata.
  Il riepilogo temporale ignora le epoche storiche prive di training_seconds.
- `models.py`: CustomResNet50 per mappe (in_channels=1) o RGB (3).
  Di default allena conv1/fc e congela BatchNorm; conv1 e sempre reinizializzata.

Configurazione: dizionario CONFIG nella cella iniziale di v2v3_ResNet50.ipynb.
Percorsi e scelte dell'esperimento restano nel notebook; config.py e rimosso.
Il notebook gestisce ciclo sulle epoche, normalizzazione, grafici e checkpoint;
usa test_step per la valutazione e normalizza la matrice di confusione per il grafico.
NEW_LR continua a prevalere sul learning rate iniziale se si esegue quella cella.

Per Colab copiare src/ml_setup nella radice Drive indicata dal notebook, oppure
impostare SOURCE_ROOT su src di un clone. Riavviare il kernel dopo aver modificato
i moduli. Anche v0_ResNet50.ipynb usa gli helper e il modello condiviso con in_channels=3.
La configurazione RGB e i percorsi restano nel suo dizionario CONFIG.


## Inferenza

v0_inference.ipynb e v2v3_inference.ipynb usano CONFIG, CustomResNet50 con
pretrained=False e test_step(..., return_predictions=True). Versione, classi
e normalizzazione restano derivate dal checkpoint. Il flag opzionale restituisce
array NumPy probabilities, targets e predictions nell'ordine del loader;
il training non li raccoglie. evaluation_seconds misura valutazione e raccolta
dei risultati, inclusi dati e trasferimenti, escludendo grafici/export.
Metriche per classe, figure e salvataggio JSON/CSV/NPZ rimangono nei notebook.
TensorBoard viene aperto manualmente; il suo export rimane opzionale.
