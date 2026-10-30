# NPZ labels

Created with `python src/build_label_manifests.py`:

- `output/v2_cumulative_rank_s1/train/manifest.json`: 50,000 samples.
- `output/v2_cumulative_rank_s1/test/manifest.json`: 10,000 samples.

Labels come from the local CIFAR-10 archive: data_batch_1..5 for train,
test_batch for test. The index restarts at zero for test; a filename without
its split is not a global identifier. `file` paths are relative to the manifest
directory. `class_to_idx` contains the ten class codes. Historical label JSON
files have not been modified. `subsample` is unchanged.

```python
import json
from pathlib import Path
import numpy as np

manifest_path = Path('output/v2_cumulative_rank_s1/train/manifest.json')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
sample = manifest['samples'][0]
with np.load(manifest_path.parent / sample['file'], allow_pickle=False) as data:
    image = data['z_u16'].astype(np.float32)[None, :, :]
label = sample['label']  # integer 0..9
# Apply the previously selected normalization here.
```

In the PyTorch Dataset, index `samples` and load each image together with its
label; return an image tensor and a torch.long label for CrossEntropyLoss.
Shuffle paired samples together. For validation, split the training manifest
records with a reproducible seed; keep the test manifest separate. For a
subsample, select records and retain their labels, updating paths if files
are copied.

Checks performed: all 60,000 filenames, all 50,000 existing labels, and PNG
content/NPZ provenance for 18 samples (the beginning, middle, and end of each
batch). This does not verify the contents of every NPZ. The script stops on
inconsistencies and does not overwrite manifests. Manifests are in output,
which is excluded from Git; the script can regenerate them. Use the pickle
script only with the trusted local CIFAR-10 archive.
