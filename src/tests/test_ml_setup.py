import ast, json, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ml_setup.helper_functions import train_step, test_step, summarize_training_time
from ml_setup.models import CustomResNet50

class MLTests(unittest.TestCase):

    def test_weighted_metrics_and_updates(self):
        torch.manual_seed(1)
        model = nn.Linear(4, 3)
        x = torch.randn(5, 4)
        y = torch.tensor([0, 1, 2, 1, 0])
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        loss = nn.CrossEntropyLoss()
        expected = loss(model(x), y).item()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = test_step(model, loader, loss, 'cpu', num_classes=3)
        self.assertAlmostEqual(result['loss'], expected, places=6)
        details = test_step(model, loader, loss, 'cpu', num_classes=3, return_predictions=True)
        self.assertEqual(details['targets'].tolist(), y.tolist())
        self.assertTrue(torch.allclose(torch.from_numpy(details['probabilities']), model(x).softmax(dim=1)))
        self.assertEqual(details['predictions'].tolist(), model(x).argmax(dim=1).tolist())
        self.assertEqual(details['confusion_counts'], result['confusion_counts'])
        self.assertGreaterEqual(details['evaluation_seconds'], 0)
        self.assertNotIn('probabilities', result)
        predictions = model(x).argmax(dim=1)
        counts = torch.zeros(3, 3, dtype=torch.long)
        for target, predicted in zip(y, predictions):
            counts[target, predicted] += 1
        self.assertEqual(result['confusion_counts'], counts.tolist())
        self.assertAlmostEqual(result['accuracy'], 100 * (predictions == y).sum().item() / len(y))
        self.assertTrue(all((torch.equal(v, before[k]) for k, v in model.state_dict().items())))
        metrics = train_step(model, loader, loss, torch.optim.SGD(model.parameters(), lr=0.1), 'cpu')
        self.assertGreaterEqual(metrics['training_seconds'], 0)
        summary = summarize_training_time([{}, metrics])
        self.assertEqual(summary['timed_epochs'], 1)
        self.assertEqual(summary['total_seconds'], metrics['training_seconds'])
        self.assertTrue(any((not torch.equal(v, before[k]) for k, v in model.state_dict().items())))
        empty = DataLoader(TensorDataset(x[:0], y[:0]))
        with self.assertRaises(ValueError):
            test_step(model, empty, loss, 'cpu')

    def test_model_compatibility(self):
        model = CustomResNet50(pretrained=False)
        restored = CustomResNet50(pretrained=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual({k for k, p in model.named_parameters() if p.requires_grad}, {'conv1.weight', 'fc.weight', 'fc.bias'})
        model.train()
        self.assertTrue(all((not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))))
        del restored, model
        rgb = CustomResNet50(pretrained=False, in_channels=3, freeze_backbone=False, freeze_batchnorm=False)
        self.assertEqual(rgb.conv1.in_channels, 3)
        rgb.train()
        self.assertTrue(rgb.bn1.training)
        self.assertTrue(all((p.requires_grad for p in rgb.parameters())))
if __name__ == '__main__':
    n = json.loads((ROOT / 'notebooks/v2v3_ResNet50.ipynb').read_text(encoding='utf-8'))
    for i, c in enumerate(n['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'cell{i}', 'exec')
    unittest.main()
