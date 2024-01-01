
class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.nn.ReLU()(self.conv1(concatenated_tensor))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(8, 8, 3, 1, 1, bias=False), Block()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], 1))
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
