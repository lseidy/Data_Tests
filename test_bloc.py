from PIL import Image
import torchvision.transforms as transforms
import torch

path = "/mnt/c/Users/lucas/Documents/LF/Lenslet_16x16_Gscale/ISO_and_Colour_Charts/ISO_Chart_16.png"
image = Image.open(path).convert('L')

transform = transforms.ToTensor()
image_tensor = transform(image)  #shape: (C, H, W), tipo: torch.FloatTensor

print(image_tensor.shape)
#image_tensor = tuple(image_tensor.shape)
#print(image_tensor) 

image_tensor = image_tensor[:,0:64,0:64]
print(image_tensor.shape) 

torch.save(image_tensor, 'test.pt')

# Converte de tensor para imagem PIL
to_pil = transforms.ToPILImage()
original = to_pil(image_tensor)  # Isso cuida da conversão automática

# Salva a imagem
original.save('Original.png')

inputBlock = image_tensor
inputBlock[:,-32:,-32:]=0
 


torch.save(image_tensor, 'train.pt')

#splitedSection = [t.unsqueeze(1)
#                  for row in torch.split(image_tensor, 16, dim=1)
#                  for t in torch.split(row, 16, dim=2)]
#
#inputBlock = torch.cat(splitedSection, dim=1)
#
#print(inputBlock.shape)
#
#torch.save(inputBlock, 'test.pt')
#
#inputBlock[:11:12,:,:] = 0
#inputBlock[:15:16,:,:] = 0
#
#torch.save(inputBlock, 'train.pt')
