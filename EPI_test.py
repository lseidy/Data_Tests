import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

import matplotlib.pyplot as plt

# Carrega a imagem como tensor
path = "/mnt/c/Users/lucas/Documents/LF/Testes/Original.png"
image = Image.open(path).convert('L')
transform = transforms.ToTensor()
image_tensor = transform(image)  # shape: (1, H, W)

# Remove a dimensão do canal
image_tensor = image_tensor.squeeze(0)  # shape: (H, W)
#image_tensor[-32:,-32:] = 0
H, W = image_tensor.shape
block_size = 16
U, V = block_size, block_size

# Resolução espacial das subimagens
h = H // U
w = W // V

# Inicializa tensor para armazenar subimagens: (U, V, h, w)
images = torch.zeros((U, V, h, w), dtype=image_tensor.dtype)

# Preenche as subimagens
for u in range(U):
    for v in range(V):
        images[u, v] = image_tensor[u::U, v::V]


save_dir = "./Block_Test"
os.makedirs(save_dir, exist_ok=True)

for u in range(U):
    for v in range(V):
        save_image(images[u, v], f"{save_dir}/img_{u:02d}_{v:02d}.png")


# Caminho da pasta
folder_path = "/mnt/c/Users/lucas/Documents/LF/Testes/Block_Test"

# Transformação para tensor
transform = transforms.ToTensor()

# Lista de tensores das imagens
image_tensors = []

# Carrega todas as imagens e transforma em tensores 2D
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path = os.path.join(folder_path, filename)
        image = Image.open(path).convert('L')  # Grayscale
        tensor = transform(image).squeeze(0)   # (H, W)
        assert tensor.shape == (4, 4), f"{filename} não tem shape (4, 4)"
        image_tensors.append(tensor)

# Empilha em um único tensor (256, 4, 4)
image_tensors = torch.stack(image_tensors)

image_grid = image_tensors.view(16, 16, 4, 4)  # (U, V, h, w)
for i in range(4):
    # Para extrair a coluna i de todas as imagens:
    epi_horizontal = image_grid[:, :, :, i]  # shape: (16, 16, 4)

# Junta as EPIs como uma imagem de shape (16*4, 16)
epi_img = epi_horizontal.permute(0, 2, 1).reshape(16*4, 16)
plt.figure(figsize=(6, 6))
plt.imshow(epi_img, cmap='gray', vmin=0, vmax=1)
plt.title("EPI Horizontal - Coluna {}".format(i))
plt.axis('off')
plt.show()

save_dir = "./EPI_H"
os.makedirs(save_dir, exist_ok=True)

#for i, line_img in enumerate(line_images):
#    save_image(line_img.unsqueeze(0), f"{save_dir}/EPI_V{i:02d}.png")  # (1, H, W)

epi_h_concat = torch.cat(line_images, dim=0)

# Se quiser salvar como imagem única:

save_image(epi_h_concat.T.unsqueeze(0), f"{save_dir}/EPI_H_concatenado_horizontalmente.png")


line_images_v = [image_stack[:, i, :] for i in range(H)] 


save_dir = "./EPI_V"
os.makedirs(save_dir, exist_ok=True)

#for i, line_img in enumerate(line_images):
#    save_image(line_img.unsqueeze(0), f"{save_dir}/EPI_H{i:02d}.png")  # (1, H, W)

epi_v_concat = torch.cat(line_images_v, dim=0)  # (N, H, 622) → (N, H, 622*1) = (N, H, 622)

# Se quiser salvar como imagem única:
save_image(epi_v_concat.unsqueeze(0), f"{save_dir}/EPI_V_concatenado_verticalmente.png")

# Caminho das pastas com as EPIs horizontais e verticais
epi_h_path = "/mnt/c/Users/lucas/Documents/LF/Testes/EPI_H/EPI_H_concatenado_horizontalmente.png"
epi_v_path = "/mnt/c/Users/lucas/Documents/LF/Testes/EPI_V/EPI_V_concatenado_verticalmente.png"

# Carregar as imagens EPI horizontais concatenadas
epi_h = Image.open(epi_h_path).convert('L')
epi_h_tensor = transforms.ToTensor()(epi_h).squeeze(0)  # (N, H)


# Extrai colunas (verticais) de cada imagem do stack (EPI horizontal)
line_images = epi_h_tensor.split(1,dim=1)

line=[]
mi=[]
for i,j in enumerate(line_images):
    line.append(j.unsqueeze(0))
    if (i+1) % 4 == 0:
        mi.append(torch.cat(line,dim=2))

        line = []

mi= torch.cat(mi,dim=0) 

# Inicializa imagem final
final_image = torch.zeros(64, 64)


# Cada bloco 16x16 será preenchido com o valor de cada pixel (i,j) das 256 imagens
for i in range(4):       # linha do pixel na imagem original 4x4
    for j in range(4):   # coluna do pixel na imagem original 4x4
        # Extrai o pixel (i,j) de todas as 256 imagens → shape: (256,)
        pixel_values = mi[:, i, j].view(16, 16)  # reshape para bloco 16x16

        # Dentro do loop
        plt.imshow(pixel_values, cmap='gray')
        plt.title(f"Pixel ({i},{j})")
        plt.colorbar()
        plt.show()
        # Define posição do bloco na imagem final
        start_h = i * 16
        start_w = j * 16

        final_image[start_h:start_h+16, start_w:start_w+16] = pixel_values



# Se quiser salvar como imagem única:
save_image(final_image, f"{save_dir}/Reconstruida_img.png")
