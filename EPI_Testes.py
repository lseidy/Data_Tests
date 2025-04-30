import torch
import matplotlib.pyplot as plt
import os
import torch
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def extract_epi(lf_tensor, direction='vertical', visualize=True, idx=0):
    """
    Extrai uma EPI e opcionalmente a visualiza.
    
    Args:
        lf_tensor: Tensor shape (B, U, V, H, W)
        direction: 'horizontal' ou 'vertical'
        visualize: se True, mostra a EPI com matplotlib
        idx: índice da amostra no batch (default 0)
    Returns:
        epi: Tensor 2D com a EPI extraída
    """
    with torch.no_grad():
        print(lf_tensor.shape)
        epis=[]
        if direction == 'horizontal':
            for y in range(lf_tensor.shape[3]):  # para todas as linhas
                epi = lf_tensor[idx, :, :, y, :]
                epis.append(epi)
            epis = torch.cat(epis, dim=2)
        elif direction == 'vertical':
            for x in range(lf_tensor.shape[4]):  # para todas as colunas
                epi = lf_tensor[idx, :, :, :, x]
                epis.append(epi)
            epis = torch.cat(epis, dim=2)
        
        #print(epis.shape)

        if visualize:
            epi_np = epis.cpu().numpy()
            plt.figure(figsize=(6, 4))
            plt.title(f"EPI ({direction})")
            if epi_np.ndim == 3:
                epi_np = epi_np[0]  # ou qualquer índice fixo para reduzir a 2D

            plt.imshow(epi_np, cmap='gray', aspect='auto')
            plt.xlabel("W" if direction == 'horizontal' else "H")
            plt.ylabel("Angular")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        return epi


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


def load_light_field_tensor(folder_path, U=16, V=16, grayscale=True):
    """
    Carrega um Light Field de uma pasta com imagens e retorna como tensor PyTorch.

    Args:
        folder_path (str): caminho da pasta com imagens (ex: lf_u0_v0.png, lf_u0_v1.png, ...)
        U (int): número de visões na horizontal (eixo u)
        V (int): número de visões na vertical (eixo v)
        grayscale (bool): se True, carrega como grayscale; se False, RGB

    Returns:
        torch.Tensor: tensor (1, U, V, H, W) se grayscale ou (1, U, V, H, W, 3) se RGB
    """
    lf = []
    for u in range(U):
        row = []
        for v in range(V):
            img = imageio.imread(folder_path)

            if grayscale:
                if img.ndim == 3:  # converter RGB para grayscale se necessário
                    img = np.mean(img, axis=2)
                img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # (H, W)
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # (H, W, 3)

            row.append(img_tensor)
        lf.append(row)

    lf_tensor = torch.stack([torch.stack(row, dim=0) for row in lf], dim=0)  # (U, V, H, W) ou (U, V, H, W, 3)
    lf_tensor = lf_tensor.unsqueeze(0)  # adiciona batch -> (1, U, V, H, W) ou (1, U, V, H, W, 3)
    return lf_tensor


#path = "/mnt/c/Users/lucas/Documents/LF/Testes/Original.png"
path = "/mnt/c/Users/lucas/Documents/LF/Lenslet_16x16_Gscale/ISO_and_Colour_Charts/ISO_Chart_16.png"

image = Image.open(path).convert('L')

transform = transforms.ToTensor()
lenslet_img = transform(image)  #shape: (C, H, W), tipo: torch.FloatTensor
# Converta
lenslet_img = lenslet_to_lf(lenslet_img, V=16, H=16)
#lf = load_light_field_tensor(path)

extract_epi(lenslet_img)