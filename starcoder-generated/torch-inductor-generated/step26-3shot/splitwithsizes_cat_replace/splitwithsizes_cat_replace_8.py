
if True:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 16, (6, 14)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 16, (6, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 16, (6, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 16, (6, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 3, (7, 15)), torch.nn.Sigmoid())
        def forward(self, v1):
            split_tensors = torch.split(v1, [1, 1, 1], dim=1)
            concatenated_tensor = torch.cat(split_tensors, dim=1)
            return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
else:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 16, (3, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 16, (6, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 16, (6, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 16, (7, 15)), torch.nn.ReLU(), torch.nn.BatchNorm2d(16), torch.nn.ConvTranspose2d(16, 3, (7, 15)), torch.nn.Sigmoid())
        def forward(self, v1):
            split_tensors = torch.split(v1, [1, 1, 1], dim=1)
            concatenated_tensor = torch.cat(split_tensors, dim=1)
            return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
v1 = torch.randn(1, 1, 16, 192)
