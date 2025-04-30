import torch
import matplotlib.pyplot as plt
import os
import torch
import imageio.v2 as imageio
import numpy as np
import torch

def extract_all_epis(lf_tensor, idx=0, direction='horizontal'):
    """
    Extrai todas as EPIs horizontais ou verticais de um light field tensor.

    Args:
        lf_tensor (torch.Tensor): Tensor com shape (batch, V, H, Y, X)
        idx (int): índice do batch a ser usado
        direction (str): 'horizontal' ou 'vertical'
    
    Returns:
        epis (list of torch.Tensor): lista com todas as EPIs
    """
    epis = []
    
    if direction == 'horizontal':
        num_y = lf_tensor.shape[3]  # número de linhas (altura)
        for y in range(num_y):
            epi = lf_tensor[idx, :, :, y, :]  # fixa y, varia x
            epis.append(epi)
    elif direction == 'vertical':
        num_x = lf_tensor.shape[4]  # número de colunas (largura)
        for x in range(num_x):
            epi = lf_tensor[idx, :, :, :, x]  # fixa x, varia y
            epis.append(epi)
    else:
        raise ValueError(f"Direção '{direction}' inválida. Use 'horizontal' ou 'vertical'.")
    
    return epis


def lenslet_to_lf(lenslet_img, V=16, H=16):
    """
    Converte uma imagem lenslet (ex: 64x64) em tensor de light field (V, H, Y, X)
    """
    B, H_full, W_full = lenslet_img.shape  # Ex: (1, 64, 64)
    subimg_Y = H_full // V  # 4
    subimg_X = W_full // H  # 4

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

path = "/mnt/c/Users/lucas/Documents/LF/Lenslet_16x16_Gscale/ISO_and_Colour_Charts/ISO_Chart_16.png"

lf = load_light_field_tensor(path)
lenslet_to_lf(lf)
#extract_all_epis(lf)
# Suponha que você já tenha seu lf_tensor carregado
#epis_horizontais = extract_all_epis(lf, idx=0, direction='horizontal')
#epis_verticais = extract_all_epis(lf, idx=0, direction='vertical')
#
#print(f"Número de EPIs horizontais extraídas: {len(epis_horizontais)}")
#print(f"Shape de uma EPI horizontal: {epis_horizontais[0].shape}")
#
#print(f"Número de EPIs verticais extraídas: {len(epis_verticais)}")
#print(f"Shape de uma EPI vertical: {epis_verticais[0].shape}")
#