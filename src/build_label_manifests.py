"""Build train/test NPZ manifests from the trusted local CIFAR-10 Python archive.

Checks every NPZ filename and any existing labels; verifies resized image content
and NPZ provenance on three deterministic samples per original CIFAR batch.
Requires NumPy and Pillow. Existing manifests are never overwritten.
"""
import argparse
import json
import pickle
import tarfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck']


def build(args):
    documents = {}
    with tarfile.open(args.archive, 'r:gz') as archive:
        for split, names in [('train', [f'data_batch_{i}' for i in range(1, 6)]),
                             ('test', ['test_batch'])]:
            folder = args.npz_root / split
            if (folder / 'manifest.json').exists():
                raise ValueError(f'Manifest already exists: {folder / "manifest.json"}')
            samples = []
            checked = []
            for name in names:
                with archive.extractfile(f'cifar-10-batches-py/{name}') as source:
                    batch = pickle.load(source, encoding='bytes')
                labels = batch[b'labels']
                if len(labels) != 10000 or len(batch[b'data']) != len(labels):
                    raise ValueError(f'Unexpected batch size: {name}')
                offset = len(samples)
                for i, label in enumerate(labels):
                    if not 0 <= label < len(CLASSES):
                        raise ValueError(f'Invalid label in {name}')
                    samples.append(dict(file=f'image_{offset+i:05d}.network.npz',
                                        label=int(label), **{'class': CLASSES[label]},
                                        cifar_batch=name, batch_index=i))
                # Match the Resize(224,224) PIL bilinear path used during export.
                for i in (0, 5000, 9999):
                    stem = f'image_{offset+i:05d}'
                    original = batch[b'data'][i].reshape(3, 32, 32).transpose(1, 2, 0)
                    expected = Image.fromarray(original).resize((224, 224), Image.Resampling.BILINEAR)
                    with Image.open(args.image_root / split / f'{stem}.png') as image:
                        if not np.array_equal(np.asarray(image.convert('RGB')), np.asarray(expected)):
                            raise ValueError(f'Image ordering/resize mismatch: {split}/{stem}')
                    with ZipFile(folder / f'{stem}.network.npz') as z:
                        meta = json.loads(z.read('metadata.json'))
                        source_name = meta['source']['path'].replace('\\', '/').split('/')[-1]
                        if source_name != f'{stem}.igd':
                            raise ValueError(f'NPZ source mismatch: {split}/{stem}')
                        # Reject a declared opposite split, while accepting older unsplit paths.
                        parts = meta['source']['path'].replace('\\', '/').split('/')[:-1]
                        if ('test' if split == 'train' else 'train') in parts:
                            raise ValueError(f'NPZ split mismatch: {split}/{stem}')
                    checked.append(stem)
            expected_names = {r['file'] for r in samples}
            actual_names = {p.name for p in folder.glob('*.npz')}
            if actual_names != expected_names:
                raise ValueError(f'{split}: missing={len(expected_names-actual_names)}, extra={len(actual_names-expected_names)}')
            # Existing user labels are an independent consistency check; preserve them.
            checked_labels = []
            expected_labels = {r['file'].replace('.network.npz', '.png'): r['label'] for r in samples}
            for path in folder.glob('*labels*.json'):
                old = json.loads(path.read_text(encoding='utf-8-sig'))
                mapping = {r['file']: r['label'] for r in old}
                if len(mapping) != len(old) or mapping != expected_labels:
                    raise ValueError(f'Existing labels disagree with official batch order: {path}')
                checked_labels.append(path.name)
            counts = {name: 0 for name in CLASSES}
            for row in samples:
                counts[row['class']] += 1
            documents[split] = dict(schema='cifar10-npz-manifest-v1', split=split,
                path_base='manifest directory', num_samples=len(samples),
                class_to_idx={name: i for i, name in enumerate(CLASSES)}, class_counts=counts,
                label_source='cifar-10-python.tar.gz / cifar-10-batches-py',
                validation=dict(all_filenames_checked=True, existing_label_files_checked=checked_labels,
                                image_content_and_npz_provenance_samples=checked,
                                scope='Content/provenance sampled; filenames and existing labels checked exhaustively'),
                samples=samples)
    # Validate both splits before writing either manifest.
    for split, document in documents.items():
        path = args.npz_root / split / 'manifest.json'
        with path.open('x', encoding='utf-8') as f:
            json.dump(document, f, indent=2)
            f.write('\n')
        print(f'{path}: {document["num_samples"]} labels; {document["class_counts"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'data/cifar-10-python.tar.gz')
    parser.add_argument('--image-root', type=Path, default=ROOT/'data/cifar-10_resized')
    parser.add_argument('--npz-root', type=Path, default=ROOT/'output/v2_cumulative_rank_s1')
    try:
        build(parser.parse_args())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error)) from error
