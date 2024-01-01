
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, 1, 0, bias=True)
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1, 1, 1], dim=1))
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1, 1, 1], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        module = [Module()]
        self.features = torch.nn.Sequential(*module * 2)
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
