import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
from argparse import Namespace
from torchsummary import summary


class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(0)  # Adiciona uma dimensão na posição 2
class RemDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(0)  # Adiciona uma dimensão na posição 2
    
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
        epis_tensor = torch.stack(epis, dim=0)
    elif direction == 'vertical':
        num_x = lf_tensor.shape[4]  # número de colunas (largura)
        for x in range(num_x):
            epi = lf_tensor[idx, :, :, :, x]  # fixa x, varia y
            epis.append(epi)
        epis_tensor = torch.stack(epis, dim=0)
    else:
        raise ValueError(f"Direção '{direction}' inválida. Use 'horizontal' ou 'vertical'.")
    
    

    return epis_tensor

class P4D(nn.Module):

    def __init__(self):
        super(P4D, self).__init__()
        n_filters = 32#params.num_filters
 
        self.pipeline = nn.Sequential( #16x16x16
            nn.Conv3d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            nn.Conv3d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.Conv3d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=3, stride=1, padding=1), nn.PReLU(),

 
        )
        self.decoder =nn.Sequential( #6,4²
            AddDimension(),
            
            nn.ConvTranspose3d(in_channels= n_filters*4, out_channels= n_filters*2, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            
            nn.ConvTranspose3d(in_channels= n_filters*2, out_channels= n_filters, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            nn.ConvTranspose3d(in_channels= n_filters, out_channels= 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            

        )
    
    def forward(self, input1, EPI_h, EPI_v):

        print("--------------------\n INPUT: ",input1.shape,EPI_h.shape, EPI_v.shape,"\n--------------------" )
        output1 = self.pipeline(input1)
        print("--------------------\n Output: ",output1.shape,"\n--------------------" )

        output = self.decoder(output1)
        print("--------------------\n Output3: ",output.shape,"\n--------------------" )

        return output
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = P4D().to(device)
model.eval()
file_path = "/mnt/c/Users/lucas/Documents/LF/Testes/train.pt"

train = torch.load(file_path).to(device)
# Defina o número de câmeras
V = 16
H = 16

# Cálculo de altura e largura de cada imagem
Y_full = train.shape[1]
X_full = train.shape[2]

# Calcula o tamanho de cada pequena imagem
subimg_Y = train.shape[1] // V
subimg_X = train.shape[2] // H

# Faz o reshape
epi_train = train.reshape(train.shape[0], V, H, subimg_Y, subimg_X)

file_path = "/mnt/c/Users/lucas/Documents/LF/Testes/test.pt"
test = torch.load(file_path).to(device)

EPI_h = extract_all_epis(epi_train, 0, 'horizontal')
EPI_v = extract_all_epis(epi_train, 0, 'vertical')

print("extracted")

lossf = nn.L1Loss()


with torch.no_grad():
    batch_size = model(train, EPI_h, EPI_v)
    rem = RemDimension()
    batch_size= rem(batch_size)
    print("batch_size: ", batch_size.shape)
    
    #summary(model, train.shape, device=str(device))

    print("loss: ", lossf(test, batch_size))


# Suponha que batch_size tenha shape (1, N, H, W)
batch_size = torch.split(batch_size, 1, dim=1)  # vira lista de (1, 1, H, W)

# Remove a dimensão do canal
batch = torch.stack([mi.squeeze(1) for mi in batch_size])  # (N, H, W)

# Divide em grupos de 8
chunks = torch.split(batch, 4)

# Junta cada grupo horizontalmente (dim=2), depois os blocos verticalmente (dim=1)
predicted_block = torch.cat(
    [torch.cat(list(chunk), dim=2) for chunk in chunks],
    dim=1
)

# Normalização [0,1]
predicted_block = (predicted_block - predicted_block.min()) / (predicted_block.max() - predicted_block.min())

print(predicted_block.shape)
# Salvar imagem
save_dir = "result_conv"
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join(save_dir, 'result_3.png')
torchvision.utils.save_image(predicted_block, fp=file_path, format="png")

print(f'Tensor salvo em {file_path}')
#