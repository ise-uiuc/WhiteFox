
class Temp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = [torch.nn.ReLU()]
        self.block2 = [torch.nn.ReLU()]
        self.block3 = [torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False), torch.nn.BatchNorm2d(64)), \
        torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False), torch.nn.BatchNorm2d(64))]
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Temp()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
