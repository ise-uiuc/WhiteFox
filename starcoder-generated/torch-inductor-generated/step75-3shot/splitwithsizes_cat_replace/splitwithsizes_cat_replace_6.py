
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 8, 8, 0, bias=False), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 8, 1, 1, 0, bias=False), torch.nn.BatchNorm2d(8), torch.nn.ReLU(inplace=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
