
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 8, 1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=False), torch.nn.ConvTranspose2d(32, 8, 1, 1), torch.nn.BatchNorm2d(8))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 2, 3, 4, 5, 6, 7, 8], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 2, 3, 4, 5, 6, 7, 8], dim=1))
# Inputs to the model
x1 = torch.randn(1, 32, 56, 56)
