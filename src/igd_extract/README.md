# Estrazione di basi, ID e posizioni dei bit da IGD

Questo programma legge un `.igd` usando la libreria Rust locale `../gdcompress`
ed esporta un JSON indentato. Non riesegue la compressione e non modifica la
repository del compressore. Il formato e le API sono quelli della copia locale
esaminata, commit `8927486e4ab16a33ab872fe1e530514653a01b4d`.

## Utilizzo

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

Ogni base e una stringa binaria a larghezza fissa. L'estrattore la ordina in
ordine crescente lessicografico, con il carattere piu a sinistra interpretato
come bit piu significativo della convenzione sperimentale. Questo evita di
convertire basi lunghe in `float` o `u64`, che ne perderebbero i bit.

Il nuovo ID e il **rango** della base ordinata. Il primo delta vale zero; ogni
successivo delta e la differenza esatta con la base di rango precedente. Il JSON
conserva sia `delta_bits` e `spatial_delta_bits` come stringhe binarie esatte,
sia due matrici normalizzate da usare come i primi due canali della rete:

```text
spatial_rank_normalized[y][x] = rank / (num_bases - 1)
spatial_delta_log2_normalized[y][x] = log2(delta + 1) / log2(max_delta + 1)
```

Il valore zero e usato per entrambi i canali quando il denominatore sarebbe
zero. `spatial_delta_bits[y][x]` non descrive la differenza dal pixel vicino:
e il delta della base associata a quel pixel nel dizionario ordinato. La
geometria resta nella posizione `(y, x)` della matrice; il significato preciso
di ciascun delta resta nel campo `sorted_dictionary`.

Le frequenze non vengono moltiplicate per i delta: sono conservate per ciascuna
base ordinata come `frequency` e coincidono con il numero di volte in cui il
suo rango appare in `spatial_rank_ids`. Moltiplicare i due valori cancellerebbe
questa distinzione.

Campi principali aggiunti dal formato `igd-network-features-v1`:

| Campo | Significato |
|---|---|
| `sorted_dictionary` | Base ordinata, ID originale, rango, delta esatto e frequenza |
| `base_min_bits` / `base_max_bits` | Estremi esatti del dizionario ordinato |
| `spatial_rank_ids` | Nuovi ID/ranghi nella geometria originale dell'immagine |
| `spatial_delta_bits` | Delta esatto associato a ogni rango, disposto spazialmente |
| `spatial_rank_normalized` | Primo canale numerico per la rete |
| `spatial_delta_log2_normalized` | Secondo canale numerico per la rete |
| `representation` | Convenzioni complete di ordine e normalizzazione |

Con gruppo `1x1`, una immagine 224x224 produce due mappe 224x224, una per
canale. I file JSON sono destinati a controllo e tracciabilita: per il training
di tutto il dataset converra poi convertire i due canali in un formato binario.

Per elaborare il dataset senza mai scrivere il JSON generico intermedio, dalla
radice del progetto eseguire prima la compressione e poi l'esportazione:

```powershell
python src/batch_compression.py
python src/batch_network_features.py
```

Il primo script usa `1x1` per default e scrive in
`data/cifar-10_compressed_pg_1x1`; il secondo legge quella cartella e scrive in
`output/network_features_pg_1x1`. Entrambi rifiutano di sovrascrivere un file
gia presente. Dopo un'interruzione, usare `--resume`. Provare prima su poche
immagini specificando `--input-dir`, `--output-dir` e, per l'esportatore,
`--verify`.

## La mappa che impedisce di mischiare i bit

**Tutti gli indici e gli ID nel JSON partono da 0.** Le stringhe di bit sono
sequenze di caratteri; non vanno interpretate come numeri binari MSB-first.

La relazione principale e:

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
il terzo in posizione 4. Le altre posizioni sono deviazioni. La lista e ordinata
per posizione crescente e le stringhe delle basi sono esportate nello stesso
ordine: non serve conoscere quale posizione l'algoritmo abbia scelto per prima.

La **cronologia delle scelte**, ad esempio `1 -> 2 -> 5 -> 3`, non e salvata
nel file IGD. `selection_order` vale quindi `null`. Non viene confusa con
l'ordine delle colonne per entropia utilizzato internamente dal delta coding:
il decoder annulla quella permutazione prima dell'esportazione. Per registrare
la cronologia effettiva bisognerebbe strumentare la fase di selezione durante
una nuova compressione.

## Campi del JSON

| Campo | Significato |
|---|---|
| `schema` | Versione dello schema di esportazione: `igd-extract-v1` |
| `source` | Percorso, dimensione, versioni IGD/EGD e codifiche del file letto |
| `conventions` | Convenzioni esplicite su indici, stringhe e disposizione |
| `image` | Dimensioni, canali, modello colore, trasformazione, raggruppamento e griglia |
| `features` | Per canale/feature: offset nel blocco, numero di bit, tipo e trasformazione |
| `chunk_bits` | Numero effettivo di bit del blocco trasformato |
| `base_bits` | Lunghezza di `bases[id].bits`, inclusi i bit costanti |
| `variable_base_bits` | Lunghezza della sola parte variabile del dizionario |
| `base_bit_positions` | Mappa colonna della base completa -> posizione nel blocco |
| `variable_base_bit_positions` | Mappa colonna di `variable_bits` -> posizione nel blocco |
| `constant_zero_bit_positions` / `constant_one_bit_positions` | Posizioni costanti reinserite nelle basi complete |
| `deviation_bit_positions` | Mappa colonna della deviazione -> posizione nel blocco |
| `bases` | Dizionario: ID, stringa completa, stringa variabile, frequenza |
| `sample_base_ids` | Un ID per gruppo di pixel, nell'ordine spaziale originale |
| `sample_deviation_bits` | Una stringa per gruppo, oppure `null` se non richiesta |
| `selection_order` | `null`: cronologia non presente nel formato |
| `verification` | Indica se e stato eseguito e superato il confronto con il decoder |

Gli ID mantengono la numerazione del dizionario nel file. Le frequenze sono
ricalcolate contando gli ID dei campioni, anche quando il dizionario e delta-coded.
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
Per la ricostruzione completa servono anche le deviazioni. Senza di esse e
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
