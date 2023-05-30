import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
y = torch.tensor([4.0, 5.0, 6.0]).to(device)
z = x + y
print(z)
