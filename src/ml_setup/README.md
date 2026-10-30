# ML helpers

- `helper_functions.py`: `train_step`, `test_step`, `summarize_training_time`.
  `train_step` returns loss, percentage accuracy, and training_seconds.
  Timing covers the epoch loop (data, transfers, forward/backward passes,
  updates); CUDA is synchronized before and after measurement.
  `test_step` returns loss, percentage accuracy, and confusion_counts as a
  list of lists (true classes in rows, predicted classes in columns).
  The number of classes is an argument. Use a scalar loss averaged per sample,
  such as unweighted CrossEntropyLoss. The timing summary ignores historical
  epochs without training_seconds.
- `models.py`: CustomResNet50 for maps (in_channels=1) or RGB (3).
  By default, it trains conv1/fc and freezes BatchNorm; conv1 is always
  reinitialized.

Configuration: the CONFIG dictionary in the initial cell of v2v3_ResNet50.ipynb.
Paths and experiment choices remain in the notebook; config.py has been removed.
The notebook manages the epoch loop, normalization, plots, and checkpoints;
it uses test_step for evaluation and normalizes the confusion matrix for plotting.
NEW_LR still overrides the initial learning rate if that cell is executed.

For Colab, copy src/ml_setup to the Drive root specified by the notebook, or
set SOURCE_ROOT to the src directory of a clone. Restart the kernel after
modifying modules. v0_ResNet50.ipynb also uses the shared helpers and model
with in_channels=3. RGB configuration and paths remain in its CONFIG dictionary.

## Inference

v0_inference.ipynb and v2v3_inference.ipynb use CONFIG, CustomResNet50 with
pretrained=False, and test_step(..., return_predictions=True). Version, classes,
and normalization are still derived from the checkpoint. The optional flag
returns NumPy probabilities, targets, and predictions arrays in loader order;
training does not collect them. evaluation_seconds measures evaluation and
result collection, including data and transfers, but excluding plots/exports.
Per-class metrics, figures, and JSON/CSV/NPZ saving remain in the notebooks.
TensorBoard is opened manually; its export remains optional.
