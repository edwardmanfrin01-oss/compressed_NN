# Review of the v2/v3 notebook
Changes relative to the received copy, with cells numbered from 1.

## Cell 1
```diff
--- before
+++ after
@@ -2,4 +2,10 @@
 In this notebook, a Image Classification model is implemented, analyzing its performance over a dataset of compressed images. The aim of the experiment is to try to make a prediction model able to classify images, starting from their compressed representation.
 
 The idea is to use the ResNet50 model (the same used as baseline on raw images dataset), modifying it according to the new input size.
-In this case, the input size consists of a single channel as input.+In this case, the input size consists of a single channel as input.
+
+**Review changes:** search for `REVIEW` in this notebook. Corrections cover
+configuration (`None`, v3 folder, version validation), NPZ validation,
+normalization cache identity, checkpoint compatibility and optional LR changes.
+The dataset formulas were corrected too. Edited code-cell outputs were cleared.
+Start a new kernel and run from the top after changing VERSION or SCALE.

```

## Cell 3
```diff
--- before
+++ after
@@ -1,14 +1,16 @@
 # 1. The dataset
 
-The CIFAR-10 dataset is a collection of images that are used to train machine learning and computer vision algorithms. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
+CIFAR-10 has 50,000 training images and 10,000 test images in 10 classes.
+The GD-derived input is one uint16 cumulative-rank map of shape [224, 224].
 
-The compressed version is obtained through the Generalized Deduplication (GD) approach. After the compression, each file has been elaborated to provide a proper shape as input for the network.
+**REVIEW: distinguish the two quantization rules.** For ordered base rank r:
 
-For each image, the output of the algorithm is a single channel, with 224x224 values.
-Each of them is obtained by:
+$$a_r=\frac{\log_2(1+\Delta_r)}{\log_2(1+\Delta_{max})}, \qquad z_r=r+\sum_{j=1}^{r}q_j.$$
 
-$$ z_i = r_i + c_i $$
-Where $c_j$ is:
-$$ c_i=∑δ_j $$
-and $δ_j$ is:
-$$ δ_i=⌊S∙\frac{log_2⁡(∆_i+1)}{log_2⁡(∆_{max}+1)}⌋ $$+- **V2:** fixed integer S; $q_r=\mathrm{round}_{half\ up}(S a_r)$.
+- **V3:** $q_r=\lfloor S_i a_r\rfloor$; choose the largest nonnegative integer
+  $S_i$ with $K_i-1+\sum_{r=1}^{K_i-1}q_r\le65535$ for each image.
+- A single-base image has an all-zero map (v3 uses $S_i=0$).
+
+Both versions use global mean/std computed only on their own training set,
+and apply the same statistics to test. No per-image rescaling is added.

```

## Cell 5
```diff
--- before
+++ after
@@ -8,7 +8,18 @@
 VERSION = "v3"
 SCALE = 1 if VERSION == "v2" else None # Set SCALE value only if v2 is selected
 USE_SUBSAMPLE = False
-CHECKPOINT_NAME = "None"  # 'None' starts a new experiment. Name of a checkpoint file (without path) resumes training from that checkpoint.
+CHECKPOINT_NAME = None  # None starts a new experiment. Name of a checkpoint file (without path) resumes training from that checkpoint.
+
+# REVIEW: reject invalid settings before selecting data or checkpoints.
+if VERSION not in ("v2", "v3"):
+    raise ValueError("VERSION must be v2 or v3")
+if VERSION == "v2" and (type(SCALE) is not int or not 0 <= SCALE <= 65535):
+    raise ValueError("V2 SCALE must be an integer in 0..65535")
+if VERSION == "v3" and SCALE is not None:
+    raise ValueError("V3 chooses its scale per image: set SCALE=None")
+# The extractor's historical name for the v2 representation ends in v1.
+REPRESENTATION = {"v2": "cumulative-rank-u16-v1", "v3": "cumulative-rank-u16-v3"}[VERSION]
+EXPERIMENT = {"version": VERSION, "representation": REPRESENTATION, "scale": SCALE}
 
 # Local notebooks may run from either the repository root or notebooks/.
 PROJECT_ROOT = Path.cwd().resolve()
@@ -21,8 +32,8 @@
     LOCAL_DATASET_NAME = f"v2_cumulative_rank-s{SCALE}"
     COLAB_DATASET_NAME = f"v2_cumulative_rank-s{SCALE}"
 elif VERSION == "v3":
-    LOCAL_DATASET_NAME = f"v3_cumulative_rank-adaptive"
-    COLAB_DATASET_NAME = f"v3_cumulative_rank-adaptive"
+    LOCAL_DATASET_NAME = f"v3_cumulative_rank_adaptive"
+    COLAB_DATASET_NAME = f"v3_cumulative_rank_adaptive"
 
 if LOCAL:
     STORAGE_ROOT = PROJECT_ROOT / "output" / LOCAL_DATASET_NAME

```

## Cell 7
```diff
--- before
+++ after
@@ -1,2 +1,16 @@
 ### 1.2 Upload all the dataset into variables
-The class `CumulativeDataset()` allows to obtain the pairs (image,label) merging the information from the compressed images and the json containing the label details+The class `CumulativeDataset()` allows to obtain the pairs (image,label) merging the information from the compressed images and the json containing the label details
+
+**REVIEW — prerequisite:** each local train/test folder must contain its label
+manifest (`manifest.json` or `manifest_train.json` / `manifest_test.json`).
+The v3 folders currently have NPZs but no manifests. From the repository root,
+the existing label builder can generate them from the trusted local CIFAR archive:
+
+```powershell
+python src/build_label_manifests.py --npz-root output/v3_cumulative_rank_adaptive
+```
+
+This command validates both splits before writing and requires the original
+CIFAR archive and resized images at its configured paths. For full Colab runs,
+copy the generated manifests to the Drive dataset root as `manifest_train.json`
+and `manifest_test.json`, alongside `train.zip` and `test.zip`.

```

## Cell 8
```diff
--- before
+++ after
@@ -2,6 +2,28 @@
 import numpy as np
 import torch
 from torch.utils.data import Dataset, DataLoader
+
+
+# REVIEW: shared validation for training, test, and normalization.
+def load_z_map(path):
+    with ZipFile(path) as archive:
+        metadata = json.loads(archive.read("metadata.json"))
+    representation = metadata.get("representation", {})
+    if representation.get("name") != REPRESENTATION:
+        raise ValueError(f"Wrong representation for {VERSION}: {path}")
+    scale = representation.get("scale")
+    if VERSION == "v2" and scale != SCALE:
+        raise ValueError(f"Wrong v2 scale: {path}")
+    if VERSION == "v3" and (
+        type(scale) is not int or not 0 <= scale <= 65535
+        or representation.get("scale_selection") != "maximum-feasible-integer"
+    ):
+        raise ValueError(f"Invalid adaptive scale metadata: {path}")
+    with np.load(path, allow_pickle=False) as archive:
+        z = archive["z_u16"]
+    if z.dtype != np.uint16 or z.shape != (224, 224):
+        raise ValueError(f"Expected uint16 [224,224]: {path}, got {z.dtype} {z.shape}")
+    return z
 
 
 class CumulativeDataset(Dataset):
@@ -25,11 +47,7 @@
     def __getitem__(self, index):
         sample = self.samples[index]
 
-        with np.load(
-            self.folder / sample["file"],
-            allow_pickle=False
-        ) as data:
-            array = data["z_u16"].astype(np.float32)
+        array = load_z_map(self.folder / sample["file"]).astype(np.float32)
 
         image = torch.from_numpy(array).unsqueeze(0)
         # shape: [1, 224, 224]

```

## Cell 9
```diff
--- before
+++ after
@@ -38,7 +38,16 @@
     if len(set(filenames)) != len(filenames):
         raise ValueError("Duplicate files in the training manifest.")
 
-    # Load cached statistics if available.
+    # REVIEW: bind cached statistics to configuration, manifest and file inventory.
+    paths = [manifest_path.parent / filename for filename in filenames]
+    fingerprint = hashlib.sha256(manifest_path.read_bytes())
+    fingerprint.update(json.dumps(EXPERIMENT, sort_keys=True).encode())
+    for path in paths:
+        stat = path.stat()
+        fingerprint.update(json.dumps([path.name, stat.st_size, stat.st_mtime_ns]).encode())
+    dataset_fingerprint = fingerprint.hexdigest()
+
+    # Load cached statistics only for the same dataset/configuration.
     if cache_path.exists() and not force:
         cached = json.loads(
             cache_path.read_text(encoding="utf-8")
@@ -53,8 +62,11 @@
                 "Invalid cached statistics; set FORCE_RECOMPUTE_STATS=True."
             )
 
-        print(f"Loaded training statistics from {cache_path}")
-        return cached
+        if (cached.get("experiment") == EXPERIMENT
+                and cached.get("dataset_fingerprint") == dataset_fingerprint):
+            print(f"Loaded training statistics from {cache_path}")
+            return cached
+        print("Statistics cache is legacy or belongs to changed data; recomputing.")
 
     paths = [
         manifest_path.parent / filename
@@ -65,16 +77,7 @@
     count, mean, m2 = 0, 0.0, 0.0
 
     for index, path in enumerate(paths, 1):
-        with np.load(path, allow_pickle=False) as archive:
-            z = archive["z_u16"]
-
-            if z.dtype != np.uint16 or z.shape != (224, 224):
-                raise ValueError(
-                    f"Expected uint16 [224,224]: {path}, "
-                    f"got {z.dtype} {z.shape}"
-                )
-
-            values = z.astype(np.float64)
+        values = load_z_map(path).astype(np.float64)
 
         n = values.size
         image_mean = float(values.mean())
@@ -106,6 +109,8 @@
 
     stats = dict(
         schema=f"z_u16-normalization-{VERSION}",
+        experiment=EXPERIMENT.copy(),
+        dataset_fingerprint=dataset_fingerprint,
         training_manifest=manifest_path.name,
         num_images=len(paths),
         num_pixels=count,

```

## Cell 20
```diff
--- before
+++ after
@@ -9,8 +9,11 @@
 checkpoint = None 
 if CHECKPOINT_TO_LOAD is not None:
     checkpoint = torch.load(CHECKPOINT_TO_LOAD, map_location="cpu", weights_only=True)
-    #if checkpoint.get("representation") != "cumulative-rank-u16-v1" or checkpoint.get("scale") < 1:
-    #    raise ValueError("Expected a v2 S=1 checkpoint created by this notebook.")
+    # REVIEW: do not silently resume another version/scale.
+    if checkpoint.get("version") != VERSION or checkpoint.get("representation") != REPRESENTATION:
+        raise ValueError("Checkpoint version/representation differs (legacy checkpoints need explicit migration).")
+    if checkpoint.get("scale") != SCALE:
+        raise ValueError("Checkpoint scale differs from the selected experiment.")
     if checkpoint["idx_to_class"] != train_data.idx_to_class:
         raise ValueError("Checkpoint class mapping differs from the current dataset.")
     for key, current in (("mean", CHANNEL_MEAN), ("std", CHANNEL_STD)):

```

## Cell 29
```diff
--- before
+++ after
@@ -1,5 +1,8 @@
-# Re-define the optimizer for changing the LR before a new round of epochs (optional)
-NEW_LR = 0.01
-optimizer = torch.optim.SGD(
-    (p for p in resnet_single_channel.parameters() if p.requires_grad), lr=NEW_LR
-)+# REVIEW: optional LR change; preserve the restored optimizer and its state.
+NEW_LR = None  # Set a positive value (e.g. 0.01) to change LR explicitly.
+if NEW_LR is not None:
+    if not math.isfinite(NEW_LR) or NEW_LR <= 0:
+        raise ValueError("NEW_LR must be finite and positive")
+    for group in optimizer.param_groups:
+        group["lr"] = NEW_LR
+print("Current learning rates:", [group["lr"] for group in optimizer.param_groups])

```

## Cell 36
```diff
--- before
+++ after
@@ -10,7 +10,7 @@
 checkpoint_to_save = {
     "schema": f"resnet-single-channel-checkpoint-{VERSION}",
     "version": VERSION,
-    "representation": f"cumulative-rank-u16-{VERSION}", "scale": SCALE,
+    "representation": REPRESENTATION, "scale": SCALE,  # REVIEW: match NPZ metadata
     "model_state_dict": resnet_single_channel.state_dict(),
     "optimizer_state_dict": optimizer.state_dict(), "num_classes": 10,
     "completed_epochs": len(history), "history": history,

```
