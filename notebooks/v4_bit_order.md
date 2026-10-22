# V4 — ordinamento MSB-first alternato tra canali

V4 mantiene la formula adattiva di v3: a_r=log2(1+d_r)/log2(1+d_max), q_r=floor(S_i*a_r), z_r=r+sum(q_0..q_r). S_i è il massimo intero ammissibile in uint16 ed è ricalcolato sui nuovi delta. Il primo delta è zero; una sola base produce una mappa zero.

L'unica modifica alla costruzione delle feature è la permutazione dei bit prima dell'ordinamento numerico del dizionario: 7,15,23,6,14,22,...,0,8,16, filtrando le posizioni selezionate. Gli indici sono zero-based e i byte originali sono LSB-first. Si permutano le colonne delle basi, si riordinano le righe, si ricalcolano rank/delta e si rimappano gli ID spaziali. Numero di basi, frequenze e associazioni tra pixel e basi sono preservati. Sono accettati tre canali da 8 bit e gruppi 1x1.

La compressione e la conversione colore non cambiano. Nei file di esempio il modello colore è YCoCgR. Non sono necessari nuovi IGD o JSON intermedi su disco. Ogni NPZ contiene z_u16 e metadata.json, inclusi ordine originale/nuovo, modello colore, numero di basi, estremi binari dopo la permutazione, massimo delta, scala e z_max.

## Uso dalla radice del progetto (PowerShell)

```powershell
cargo build --release --manifest-path src/igd_extract/Cargo.toml --target-dir src/gdcompress/target --bin igd_extract
python src/batch_network_features.py --version v4 --input-dir data/cifar-10_compressed_1x1/train --output-dir output/v4_cumulative_rank_interleaved/train
python src/batch_network_features.py --version v4 --input-dir data/cifar-10_compressed_1x1/test --output-dir output/v4_cumulative_rank_interleaved/test
```

Adattare i percorsi alle cartelle effettive. Il batch legge gli IGD direttamente nella cartella indicata, senza ricorsione. Aggiungere --verify per controllare i chunk decodificati e --resume per riprendere un'esportazione della stessa versione. Produce scales_v4.csv e summary_v4.jsonl (una riga per immagine) per ogni cartella elaborata; i riepiloghi sono ricostruiti anche in resume e sostituiti solo al termine. Nessun dizionario completo o matrice spaziale viene duplicato nel riepilogo.

Per un file singolo: igd_extract INPUT.igd --representation cumulative-rank-interleaved -o OUTPUT.npz.

I notebook v2v3_ResNet50, valid_v2v3_ResNet50 e v2v3_inference supportano v4. Impostare CONFIG["version"]="v4", scale=None e checkpoint_name=None nel training, mantenendo i manifest di etichette secondo la procedura esistente. Usare un nuovo kernel e statistiche calcolate sul training v4; non riutilizzare statistiche/checkpoint v3. L'architettura del modello non cambia. Il default dei notebook resta v3.

## Esperimenti successivi (non implementati)

- MSB-first per canale senza alternanza: 7,6,...,0,15,14,...,8,23,22,...,16.
- Confronto RGB / YCoCgR / YCbCr con trasformazioni esplicite e documentate, evitando di applicare YCoCgR a valori YCbCr impacchettati come RGB.

L'ordine attuale delle versioni precedenti rimane disponibile. Non è stato avviato un nuovo training né esportato l'intero dataset.
