import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
# Caminho da pasta
folder_path = "/mnt/c/Users/lucas/Documents/LF/Testes/imagens_extraidas"

# Transformação para tensor
transform = transforms.ToTensor()

# Lista de tensores das imagens
image_tensors = []

# Carrega todas as imagens da pasta
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path = os.path.join(folder_path, filename)
        image = Image.open(path).convert('L')  # grayscale
        tensor = transform(image).squeeze(0)  # shape: (432, 622)
        assert tensor.shape == (432, 622), f"Imagem {filename} não tem shape 432x622!"
        image_tensors.append(tensor)

# Stack para criar tensor (N, 432, 622)
image_stack = torch.stack(image_tensors)  # shape: (N, 16, 16)

# Separa por linha: resultado será lista de 16 imagens (cada uma de shape (N, 16))
line_images = [image_stack[:, :, i] for i in range(622)]  # 16 tensores (N, 16)


save_dir = "./EPI_H"
os.makedirs(save_dir, exist_ok=True)

for i, line_img in enumerate(line_images):
    save_image(line_img.unsqueeze(0), f"{save_dir}/linha_{i:02d}.png")  # (1, H, W)
