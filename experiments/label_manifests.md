# Etichette degli NPZ

Creati con `python src/build_label_manifests.py`:

- `output/v2_cumulative_rank_s1/train/manifest.json`: 50.000 campioni.
- `output/v2_cumulative_rank_s1/test/manifest.json`: 10.000 campioni.

Le etichette provengono dall'archivio CIFAR-10 locale: data_batch_1..5 per
train, test_batch per test. L'indice riparte da zero nel test; il nome del
file senza split non è un identificatore globale. I percorsi `file` sono
relativi alla cartella del manifest. `class_to_idx` contiene i dieci codici.
I JSON storici di etichette non sono stati modificati. `subsample` è intatto.

```python
import json
from pathlib import Path
import numpy as np

manifest_path = Path('output/v2_cumulative_rank_s1/train/manifest.json')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
sample = manifest['samples'][0]
with np.load(manifest_path.parent / sample['file'], allow_pickle=False) as data:
    image = data['z_u16'].astype(np.float32)[None, :, :]
label = sample['label']  # intero 0..9
# Applicare qui la normalizzazione già scelta.
```

Nel Dataset PyTorch usare l'indice su `samples`, caricando insieme immagine
ed etichetta; restituire un tensore immagine e label torch.long per
CrossEntropyLoss. Lo shuffle va applicato ai campioni accoppiati. Per la
validation suddividere i record del manifest train con un seed riproducibile;
il manifest test resta separato. Per subsample scegliere record e conservare
le rispettive etichette, aggiornando i percorsi se vengono copiati i file.

Verifiche effettuate: tutti i 60.000 nomi di file, tutte le 50.000 etichette
preesistenti e contenuto PNG/provenienza NPZ su 18 campioni (inizio, metà e
fine di ciascun batch). Questo non equivale a verificare il contenuto di
tutti gli NPZ. Lo script si arresta su incongruenze e non sovrascrive manifest.
I manifest sono in output, escluso da Git; lo script permette di rigenerarli.
Lo script pickle va usato solo con l'archivio CIFAR-10 locale attendibile.
