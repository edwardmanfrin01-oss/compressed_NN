# Estrazione di basi, ID e posizioni dei bit da IGD

Questo programma legge un `.igd` usando la libreria Rust locale `../gdcompress`
ed esporta un JSON indentato. Non riesegue la compressione e non modifica la
repository del compressore. Il formato e le API sono quelli della copia locale
esaminata, commit `8927486e4ab16a33ab872fe1e530514653a01b4d`.


## Utilizzo (this command below shows why the executable of igd_extract is in the gdcompress project folder)

Eseguire dalla cartella del progetto `C:\Users\edward\CODING_PROJECTS\AARHUS`:

```powershell
cargo run --release --locked --manifest-path src/igd_extract/Cargo.toml --target-dir src/gdcompress/target -- data/cifar-10_compressed/image_00000.igd -o output/igd/image_00000.json --verify
```

Dopo la compilazione si puo usare direttamente l'eseguibile:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed/image_00001.igd -o output/igd/image_00001.json --verify
```

Opzioni:

- `-o PATH` / `--output PATH`: destinazione; il default e `INPUT.decoded.json`.
- `--network-features`: genera direttamente la rappresentazione del primo
  esperimento: dizionario ordinato, delta esatti e mappe spaziali di rango e delta.
- `--network-npz`: directly generates the .npz file containing the two spatial maps 
  (`spatial_ranks` and `spatial_deltas`), ready as input for the Model. Values 
  are normalized and mapped in [0,255] to obain a uint8 representation (see below). 
- `--verify`: ricostruisce i blocchi trasformati da basi, ID, mappa e deviazioni
  e confronta **ogni bit di ogni campione** con il decoder di `gdcompress`.
  Non crea un'immagine RGB; questa verifica richiede lavoro e memoria aggiuntivi.
- `--include-deviations`: salva anche le deviazioni, necessarie per ricostruire
  integralmente i blocchi trasformati dal solo JSON.
- `--help`: riepilogo.

Le cartelle di destinazione sono create se necessario. Un output gia esistente
non viene sovrascritto. Un errore produce un messaggio su stderr e codice di
uscita 1. Il programma controlla la lunghezza del contenitore prima di passarlo
al parser: un file interrotto durante la scrittura puo essere incompleto.

Per includere le deviazioni, usare un nome diverso se il primo output esiste:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed/image_00000.igd -o output/igd/image_00000.with_deviations.json --include-deviations --verify
```

## Rappresentazione per il primo esperimento di rete

`--network-features` legge direttamente l'IGD e scrive un JSON separato; il JSON
generico non viene salvato come file intermedio. Esempio per immagini compresse
con `--pixel-grouping 1x1`:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed/image_00000.igd -o output/network/image_00000.network.json --network-features --verify
```

Ogni base è una stringa binaria a larghezza fissa. L'estrattore la ordina in
ordine crescente lessicografico, con il carattere piu a sinistra interpretato
come bit piu significativo della convenzione sperimentale. Questo evita di
convertire basi lunghe in `float` o `u64`, che ne perderebbero i bit.

Il nuovo ID è il **rango** della base ordinata. Il primo delta vale zero; ogni
successivo delta è la differenza esatta con la base di rango precedente. Il JSON
conserva sia `delta_bits` e `spatial_delta_bits` come stringhe binarie esatte,
sia due matrici normalizzate da usare come i primi due canali della rete:

```text
spatial_rank_normalized[y][x] = rank / (num_bases - 1)
spatial_delta_log2_normalized[y][x] = log2(delta + 1) / log2(max_delta + 1)
```

Il valore zero è usato per entrambi i canali quando il denominatore sarebbe
zero. `spatial_delta_bits[y][x]` non descrive la differenza dal pixel vicino:
è il delta della base associata a quel pixel nel dizionario ordinato. La
geometria resta nella posizione `(y, x)` della matrice; il significato preciso
di ciascun delta resta nel campo `sorted_dictionary`.

Le frequenze non vengono moltiplicate per i delta: sono conservate per ciascuna
base ordinata come `frequency` e coincidono con il numero di volte in cui il
suo rango appare in `spatial_rank_ids`. Moltiplicare i due valori cancellerebbe
questa distinzione.

Campi principali aggiunti dal formato `igd-network-features-v1`:

| Campo                             | Significato |
|---                                |---|
| `sorted_dictionary`               | Base ordinata, ID originale, rango, delta esatto e frequenza |
| `base_min_bits` / `base_max_bits` | Estremi esatti del dizionario ordinato |
| `spatial_rank_ids`                | Nuovi ID/ranghi nella geometria originale dell'immagine |
| `spatial_delta_bits`              | Delta esatto associato a ogni rango, disposto spazialmente |
| `spatial_rank_normalized`         | Primo canale numerico per la rete |
| `spatial_delta_log2_normalized`   | Secondo canale numerico per la rete |
| `representation`                  | Convenzioni complete di ordine e normalizzazione |

Con gruppo `1x1`, una immagine 224x224 produce due mappe 224x224, una per
canale. I file JSON sono destinati a controllo e tracciabilita: per il training
di tutto il dataset usare `--network-npz`, che scrive direttamente due canali
`uint8` senza creare un JSON intermedio:

```powershell
.\src\gdcompress\target\release\igd_extract.exe data/cifar-10_compressed_pg_1x1/image_00000.igd -o output/network/image_00000.network.npz --network-npz --verify
```

La conversione in `uint8` viene fatta soltanto dopo la normalizzazione:

```text
rank_u8  = round(255 * spatial_rank_normalized)
delta_u8 = round(255 * spatial_delta_log2_normalized)
```

Il loader di PyTorch riporta approssimativamente i due canali in `[0, 1]` con
`array.astype(float32) / 255.0`. L'NPZ contiene `rank_u8.npy` e `delta_u8.npy`,
entrambi di forma `(height, width)`, piu `metadata.json`. Per una immagine
224x224 i soli tensori occupano esattamente `224 * 224 * 2 = 100352` byte.

Per elaborare il dataset senza mai scrivere il JSON generico intermedio, dalla
radice del progetto eseguire prima la compressione e poi l'esportazione:

```powershell
python src/batch_compression.py
python src/batch_network_features.py
```

Il primo script usa `1x1` per default e scrive in
`data/cifar-10_compressed_pg_1x1`; il secondo legge quella cartella e scrive in
`output/network_features_1x1_v1`. Entrambi rifiutano di sovrascrivere un file
gia presente. Dopo un'interruzione, usare `--resume`. Provare prima su poche
immagini specificando `--input-dir`, `--output-dir` e, per l'esportatore,
`--verify`.

In alternativa, `python src/build_1x1_npz_dataset.py` esegue nella stessa
iterazione PNG -> IGD 1x1 -> NPZ uint8 e conserva l'IGD. Non crea JSON per il
dataset; usare `--verify` solo per un piccolo campione iniziale e `--resume`
per riprendere dopo un'interruzione.

## La mappa che impedisce di mischiare i bit

**Tutti gli indici e gli ID nel JSON partono da 0.** Le stringhe di bit sono
sequenze di caratteri; non vanno interpretate come numeri binari MSB-first.

La relazione principale è:

```text
blocco_trasformato[base_bit_positions[k]] = bases[id].bits[k]
```

Esempio schematico:

```json
{
  "base_bit_positions": [0, 1, 4],
  "bases": [{"id": 0, "bits": "101"}],
  "sample_base_ids": [0]
}
```

Il primo carattere di `101` va in posizione 0, il secondo in posizione 1,
il terzo in posizione 4. Le altre posizioni sono deviazioni. La lista è ordinata
per posizione crescente e le stringhe delle basi sono esportate nello stesso
ordine: non serve conoscere quale posizione l'algoritmo abbia scelto per prima.

La **cronologia delle scelte**, ad esempio `1 -> 2 -> 5 -> 3`, non è salvata
nel file IGD. `selection_order` vale quindi `null`. Non viene confusa con
l'ordine delle colonne per entropia utilizzato internamente dal delta coding:
il decoder annulla quella permutazione prima dell'esportazione. Per registrare
la cronologia effettiva bisognerebbe strumentare la fase di selezione durante
una nuova compressione.

## Campi del JSON

| Campo                         | Significato |
|---                            |---|
| `schema`                      | Versione dello schema di esportazione: `igd-extract-v1` |
| `source`                      | Percorso, dimensione, versioni IGD/EGD e codifiche del file letto |
| `conventions`                 | Convenzioni esplicite su indici, stringhe e disposizione |
| `image`                       | Dimensioni, canali, modello colore, trasformazione, raggruppamento e griglia |
| `features`                    | Per canale/feature: offset nel blocco, numero di bit, tipo e trasformazione |
| `chunk_bits`                  | Numero effettivo di bit del blocco trasformato |
| `base_bits`                   | Lunghezza di `bases[id].bits`, inclusi i bit costanti |
| `variable_base_bits`          | Lunghezza della sola parte variabile del dizionario |
| `base_bit_positions`          | Mappa colonna della base completa -> posizione nel blocco |
| `variable_base_bit_positions` | Mappa colonna di `variable_bits` -> posizione nel blocco |
| `constant_zero_bit_positions` | Posizioni costanti (with value 0) reinserite nelle basi complete | 
| `constant_one_bit_positions`  | Posizioni costanti (with value 1) reinserite nelle basi complete |
| `deviation_bit_positions`     | Mappa colonna della deviazione -> posizione nel blocco |
| `bases`                       | Dizionario: ID, stringa completa, stringa variabile, frequenza |
| `sample_base_ids`             | Un ID per gruppo di pixel, nell'ordine spaziale originale |
| `sample_deviation_bits`       | Una stringa per gruppo, oppure `null` se non richiesta |
| `selection_order`             | `null`: cronologia non presente nel formato |
| `verification`                | Indica se e stato eseguito e superato il confronto con il decoder |

Gli ID mantengono la numerazione del dizionario nel file. Le frequenze sono
ricalcolate contando gli ID dei campioni, anche quando il dizionario è delta-coded.
Il numero di basi e la loro lunghezza possono variare da immagine a immagine.
Un ID ha significato solo insieme al dizionario della stessa immagine.

La griglia si percorre da sinistra a destra e dall'alto verso il basso:

```text
group_x = sample_index % image.grid_width
group_y = sample_index // image.grid_width
pixel_x_inizio = group_x * image.pixel_group_width
pixel_y_inizio = group_y * image.pixel_group_height
```

I gruppi ai bordi possono estendersi oltre le dimensioni dell'immagine: queste
dimensioni rimangono nel JSON per consentire il ritaglio corretto. L'estrattore
rifiuta campioni condensati con pesi, che non forniscono una mappa spaziale
originale affidabile.

## Attenzione: blocco RGB e blocco trasformato

Per RGB8, un gruppo grezzo 3x3 contiene `3 * 9 * 8 = 216` bit. Con la
trasformazione `ForFirstPixel`, per ogni canale si salvano il primo valore
su 8 bit e 8 differenze codificate su 9 bit:

```text
3 * (8 + 8 * 9) = 240 bit
```

Questo e il caso del file reale `image_00000.igd` esaminato. Quindi le posizioni
si riferiscono a un blocco trasformato di 240 bit, con tre feature da 80 bit,
non a 216 bit RGB. I bit sono raggruppati per canale; i campi numerici sono
memorizzati dal bit meno significativo. Per ottenere i pixel RGB bisogna prima
ricomporre il blocco e poi invertire le trasformazioni di gruppo e colore usando
i metadati e le funzioni di `gdcompress`.

**Basi + ID + posizioni non contengono generalmente tutti i valori originali.**
Per la ricostruzione completa servono anche le deviazioni. Senza di esse è
possibile identificare i bit noti, ma non bisogna sostituire i bit mancanti con
zeri fingendo che il risultato sia la ricostruzione originale. Il JSON non
recupera nemmeno informazioni perse prima della compressione, per esempio in
un ridimensionamento dell'immagine.

## Lettura in Python

Solo libreria standard; da eseguire dalla cartella AARHUS:

```python
import json
from pathlib import Path

data = json.loads(Path("output/igd/image_00000.json").read_text(encoding="utf-8"))
sample_index = 0
base_id = data["sample_base_ids"][sample_index]
base = data["bases"][base_id]

# Mappa completa del blocco, con None dove mancano le deviazioni.
chunk = [None] * data["chunk_bits"]
for position, bit in zip(data["base_bit_positions"], base["bits"], strict=True):
    chunk[position] = int(bit)

if data["includes_deviations"]:
    deviation = data["sample_deviation_bits"][sample_index]
    for position, bit in zip(data["deviation_bit_positions"], deviation, strict=True):
        chunk[position] = int(bit)
    assert None not in chunk

print("ID:", base_id)
print("Base:", base["bits"])
print("Posizioni:", data["base_bit_positions"])
print("Blocco trasformato:", chunk)
```

Per NumPy, gli ID si possono disporre sulla griglia con
`np.asarray(data['sample_base_ids']).reshape(data['image']['grid_height'], data['image']['grid_width'])`.
Questa e una griglia di **ID locali ai gruppi**, non una normale immagine RGB.

## Verifiche del codice

```powershell
cargo test --release --locked --manifest-path src/igd_extract/Cargo.toml --target-dir src/gdcompress/target
```

I test confrontano i blocchi ricostruiti dal JSON serializzato con i dati prima
della compressione: codifiche normal/RLE/Huffman, dizionari raw/delta-unary/
delta-fixed, trasformazioni Raw/ForFirstPixel/ForMin, bit costanti 0 e 1,
selezione in ordine non crescente e gruppi ai bordi. Sono controllati anche
file troncati e rifiuto della sovrascrittura.

Il `Cargo.lock` mantiene le versioni delle dipendenze risolte nella copia locale;
conservarlo per riprodurre la compilazione. Il JSON privilegia la leggibilita e
puo essere molto piu grande dell'IGD, soprattutto includendo le deviazioni.
Non e ancora il formato ottimizzato di input per una rete neurale.
# Modalità cumulativa + rango (`uint16`)

## Conversione in batch con versioni

Dalla radice del progetto:

```powershell
python src/batch_network_features.py --version v1 --resume
python src/batch_network_features.py --version v2 --scale 1 --resume
```

L'input predefinito è `data/cifar-10_compressed_1x1`. V1 salva i due canali
uint8 in `output/network_features_1x1_v1`; v2 salva `z_u16` in
`output/cumulative_rank_s1` (il nome cambia con S). Senza `--version` si usa
v1; per v2 S vale 1 se omesso. `--scale` non è valido con v1.
Si possono specificare `--input-dir`, `--output-dir`, `--extractor-bin` e
`--verify`. Entrambe le versioni mantengono i nomi `image_XXXXX.network.npz`.

`--resume` controlla integrità ZIP, array richiesti, rappresentazione e S
prima di saltare un output esistente; con `--verify` richiede anche che
l'output precedente sia stato verificato. Non confronta il contenuto del
file IGD con quello usato in precedenza: dopo una ricompressione usare una
nuova cartella. Il primo errore interrompe il batch e ne mostra il motivo;
i file già completati possono essere riutilizzati con `--resume`.

Questa modalità mantiene disponibile il precedente output `--network-npz`
a due canali. Per il nuovo esperimento, dalla radice di AARHUS:

```powershell
cargo build --manifest-path src/igd_extract/Cargo.toml --bin igd_extract --target-dir src/gdcompress/target --release --offline --locked
src/gdcompress/target/release/igd_extract.exe data/cifar-10_compressed_1x1/image_00000.igd --representation cumulative-rank --scale 1 --verify -o output/cumulative_rank_s1/image_00000.npz
```

`--scale` è un intero in 0..65535, default 1, utilizzabile solo con
`--representation cumulative-rank`. S=0 produce il solo rango. Sono richiesti
gruppi 1x1. Senza `-o`, il nome è `INPUT.cumulative-s1.npz` (con S nel nome).
Un file esistente non viene sovrascritto. L'overflow uint16 interrompe
l'operazione prima di creare il file: non viene applicato clipping.

Per le basi ordinate si calcola `d[0]=0`, `d[r]=base[r]-base[r-1]`, poi
`q[r]=round(S*log2(1+d[r])/log2(1+d_max))`, con arrotondamento dei mezzi verso
l'alto. Se d_max=0 tutti i q sono zero. La somma cumulativa dei q viene
aggiunta al rango originale, non normalizzato. Infine ogni pixel riceve
il valore corrispondente alla propria base. Non si pesano i gap per frequenza.
La convenzione numerica sulle basi è la stessa della versione a due canali.

L'archivio NPZ contiene `z_u16.npy` (un array uint16 little-endian H×W) e
`metadata.json`: S, formula, convenzioni, dimensioni, numero di basi,
cumulativa finale, z massimo, sorgente e verifica. Si usa ZIP senza ulteriore
compressione. Non vengono scritti JSON intermedi su disco; i dati intermedi
sono costruiti in RAM per la singola immagine. `--verify` verifica i blocchi
trasformati decodificati, non l'accuracy della rappresentazione.

```python
import json
import numpy as np
from zipfile import ZipFile

path = 'output/cumulative_rank_s1/image_00000.npz'
with np.load(path, allow_pickle=False) as data:
    z = data['z_u16']                       # (H, W), uint16
    x = z.astype(np.float32)[None] / 65535  # (1, H, W), solo al caricamento
with ZipFile(path) as archive:
    metadata = json.loads(archive.read('metadata.json'))
```

Da Python si può richiamare lo stesso comando con `subprocess.run([...],
check=True)`. La generazione in batch e il training non vengono avviati
automaticamente. Tenere cartelle distinte per S e per rappresentazione.
