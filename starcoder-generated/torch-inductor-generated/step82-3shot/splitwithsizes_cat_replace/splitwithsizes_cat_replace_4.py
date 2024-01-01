
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.nn.ReLU()(self.conv1(concatenated_tensor))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(3, 32, 3, 1, 1)]
        block_1 = [Block()]
        block_2 = [Block()]
        block_3 = [Block()]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
