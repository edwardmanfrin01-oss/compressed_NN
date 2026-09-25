import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# 1. Carica l'immagine con PIL in modalità RGB
img_pil_rgb = Image.open('data/cifar-10_resized/image_00000.png').convert('RGB')

# 2. Converti l'immagine PIL nello spazio colore YCbCr
img_pil_ycbcr = img_pil_rgb.convert('YCbCr')

# 3. Converti il risultato in un tensore PyTorch tramite torchvision
# Il tensore avrà dimensioni [Canali, Altezza, Larghezza] -> [3, H, W]
tensor_ycbcr = F.to_tensor(img_pil_ycbcr)

# 4. Separa i canali del tensore
Y_tensor = tensor_ycbcr[0, :, :]
Cb_tensor = tensor_ycbcr[1, :, :]
Cr_tensor = tensor_ycbcr[2, :, :]

print("Formato tensore YCbCr:", tensor_ycbcr.shape)

# 1. Convertiamo i tensori in array NumPy per Matplotlib
# Usiamo .cpu() nel caso in cui i tuoi tensori si trovino su una GPU (CUDA/MPS)
Y_np = Y_tensor.cpu().numpy()
Cb_np = Cb_tensor.cpu().numpy()
Cr_np = Cr_tensor.cpu().numpy()

# 2. Configuriamo la griglia di visualizzazione (1 riga, 3 colonne)
plt.figure(figsize=(15, 5))

# Canale Y (Luminanza) - Usiamo la mappa di colore 'gray' perché è in bianco e nero
plt.subplot(1, 3, 1)
plt.imshow(Y_np, cmap='gray')
plt.title('Canale Y (Luminanza)')
plt.axis('off')  # Nasconde gli assi con i pixel per una visualizzazione pulita

# Canale Cb (Crominanza Blu)
plt.subplot(1, 3, 2)
plt.imshow(Cb_np, cmap='coolwarm')  # 'coolwarm' evidenzia bene le transizioni tra freddo/caldo (Giallo/Blu)
plt.title('Canale Cb (Blu-Giallo)')
plt.axis('off')

# Canale Cr (Crominanza Rosso)
plt.subplot(1, 3, 3)
plt.imshow(Cr_np, cmap='coolwarm')  # Evidenzia le transizioni Rosso/Verde
plt.title('Canale Cr (Rosso-Verde)')
plt.axis('off')

# 3. Mostra la finestra con le immagini
plt.tight_layout()
plt.show()
