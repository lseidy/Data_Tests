import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def lenslet_to_lf(lenslet_img, V=16, H=16):
    """
    Converte imagem lenslet (64x64) em tensor Light Field de shape (B, V, H, Y, X)
    """
    B, H_full, W_full = lenslet_img.shape
    subimg_Y = H_full // V
    subimg_X = W_full // H

    lf = torch.zeros((B, V, H, subimg_Y, subimg_X), dtype=lenslet_img.dtype)

    for v in range(V):
        for h in range(H):
            lf[:, v, h, :, :] = lenslet_img[:, v::V, h::H]

    return lf

# Exemplo: carregue sua imagem lenslet aqui
# Suponha que já seja um tensor torch com shape (1, 64, 64)
# Substitua isso com torch.load(...) se estiver carregando de arquivo
path = "/mnt/c/Users/lucas/Documents/LF/Lenslet_16x16_Gscale/ISO_and_Colour_Charts/ISO_Chart_16.png"

image = Image.open(path).convert('L')

transform = transforms.ToTensor()
lenslet_img = transform(image)  #shape: (C, H, W), tipo: torch.FloatTensor
# Converta
lenslet_img = lenslet_to_lf(lenslet_img, V=16, H=16)

B, V, H, Y, X = lenslet_img.shape
canvas = torch.zeros((V * Y, H * X))
for v in range(V):
    for h in range(H):
        canvas[v*Y:(v+1)*Y, h*X:(h+1)*X] = lenslet_img[0, v, h]
plt.figure(figsize=(10, 10))
plt.imshow(canvas.numpy(), cmap='gray')
plt.axis('off')
plt.title('Visualização Multiview (Grade V×H)')
plt.show()