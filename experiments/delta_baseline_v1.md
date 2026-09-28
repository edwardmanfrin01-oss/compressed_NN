# Baseline senza geometria: delta del dizionario

## Obiettivo

Verificare con un esperimento minimo se i delta tra basi ordinate contengono
informazioni utili per classificare immagini CIFAR-10. Questa baseline usa solo
i delta. Non include ID spaziali, frequenze, basi minime, deviazioni o maschere.
Non e una rappresentazione lossless dell'immagine e non usa ResNet50.

## Procedura eseguita il 28 settembre 2026

1. Selezione casuale bilanciata di 40 PNG per classe dai file di training gia
   ridimensionati a 224x224. Etichette: `data/cifar-10_labels.json`. Seed: 42.
2. Suddivisione per immagine e per classe: 32 train + 8 validation. Totali:
   320 train e 80 validation. Non e il test set ufficiale di CIFAR-10.
3. Compressione temporanea di ciascuna PNG con `--pixel-grouping 1x1`, tramite
   il compressore esistente. Restano le trasformazioni colore del compressore.
4. Lettura con l'estrattore Rust esistente; il JSON generico e solo temporaneo.
5. Interpretazione della stringa di ogni base come intero MSB-first, identica
   alla convenzione dell'esperimento precedente. Questa convenzione non equivale
   a un'intensita RGB e non indica l'ordine di selezione per entropia.
6. Ordinamento crescente delle basi e sottrazione esatta tra consecutive:
   K basi producono K-1 delta. I delta NON vengono riordinati dopo la sottrazione.
7. Conversione numerica `log2(1 + delta) / 24`. Il denominatore e fisso, non
   stimato sul dataset. I delta esatti rimangono nel file come `uint32`.
8. Divisione dell'asse della sequenza in 256 intervalli uguali e media per
   intervallo (con pesi frazionari ai bordi). Per sequenze corte i valori vengono
   distribuiti su piu intervalli, senza aggiungere zeri. Una sola base produce
   convenzionalmente un vettore nullo, perche non esistono differenze tra basi.
9. Standardizzazione delle 256 feature usando media/deviazione del SOLO training
   set, clipping a [-10,10], MLP 256 -> 64 ReLU -> 10 softmax. Adam, learning
   rate 0.001, L2 sui pesi 0.001, batch 32, 30 epoche, seed 42.

La media riduce il dettaglio delle sequenze e non conserva il numero di basi.
Si conserva l'ordine lungo il dizionario, ma non la sequenza originale completa.
La geometria dell'immagine non e presente: immagini con lo stesso dizionario
producono lo stesso input anche se la disposizione dei pixel e diversa.

## Risultato pilota

| Quantita | Risultato |
|---|---:|
| Immagini training / validation | 320 / 80 |
| Parametri del modello | 17.098 |
| Epoche | 30 |
| Accuracy training finale | 82,8125% |
| Accuracy validation finale e migliore | 18,75% (15/80) |
| Riferimento casuale uniforme | 10% |
| Numero delta per immagine, min / mediana / max | 1 / 887,5 / 18.401 |
| Bit selezionati per base, min / max | 1 / 24 |
| Dataset NPZ, inclusi delta esatti e metadati | 679.850 byte |
| Input del modello per immagine | 256 float32 = 1.024 byte |

Il grande divario train/validation indica overfitting. Con 80 immagini di
validation non si puo concludere che la rappresentazione generalizzi bene.
La migliore epoca e scelta sulla validation; non e una misura indipendente
di test e non e un confronto controllato con ResNet50.

La preparazione ha richiesto circa 170 secondi su questo computer. La rete e
molto piccola e il training e durato meno di un secondo (escluso avvio/import).
Tempi indicativi, non un benchmark.

## File e comandi

- Codice: `src/delta_baseline.py` (NumPy, nessun PyTorch necessario).
- Test: `src/test_delta_baseline.py`.
- Dati prodotti: `output/delta_baseline/pilot.npz`.
- Metriche, configurazione e liste degli split: `runs/delta_baseline/pilot/report.json`.
- Modello alla migliore validation: `runs/delta_baseline/pilot/best_model.npz`.

I file prodotti sono gia presenti; per ripetere, usare nomi nuovi. Dalla radice
AARHUS, con Python + NumPy disponibili:

```powershell
python src/delta_baseline.py prepare --per-class 40 --output output/delta_baseline/pilot_v2.npz
python src/delta_baseline.py train --data output/delta_baseline/pilot_v2.npz --run-dir runs/delta_baseline/pilot_v2
python -m unittest discover -s src -p test_delta_baseline.py
```

Nel terminale usato qui `python` non e nel PATH. Per usare il runtime con cui
e stata effettuata la prova:

```powershell
$deltaPython = 'C:/Users/edward/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $deltaPython src/delta_baseline.py --help
```

Sostituire `python` nei comandi precedenti con `& $deltaPython`.

Il dataset salva `X` (N,256), `y`, `filenames`, `validation`, `base_bits`,
`delta_values` e `delta_offsets`, oltre ai metadati. Per l'immagine i, i delta
originali sono `delta_values[delta_offsets[i]:delta_offsets[i+1]]`. Non servono
maschere o padding per conservarli. Usare `np.load(path, allow_pickle=False)`.
Il file salva anche i commit dei repository come contesto; le modifiche locali
non committate non sono rappresentate dal solo hash.

## Dimensioni: confronti corretti

Il vettore di 256 float32 e una sintesi con perdita, non una nuova compressione
lossless. Un'immagine RGB 224x224 grezza occupa 150.528 byte, la CIFAR originale
32x32 ne occupa 3.072. Il vettore occupa 1.024 byte, esclusi etichetta/metadati.
La sequenza esatta richiede invece `4*(K-1)` byte in uint32 e varia per immagine.
L'NPZ usa inoltre compressione ZIP, quindi il peso del file va misurato.

Per decidere se proseguire: aumentare il campione mantenendo lo split fisso,
confrontare piu seed e confrontare con un controllo (per esempio solo numero
di basi o un istogramma dei delta). I delta esatti gia salvati permettono di
cambiare il riassunto senza ricomprimere le immagini di questo campione.
