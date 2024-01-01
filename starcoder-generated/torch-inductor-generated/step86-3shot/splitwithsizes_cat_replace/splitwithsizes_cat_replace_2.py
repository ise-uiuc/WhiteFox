
class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return self.bn1(self.conv1(torch.cat(split_tensors, dim=1)))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(32)]
        block_1 = [torch.nn.ReLU()]
        block_2 = [torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)]
        block_3 = [Block1()]
        block_4 = [torch.nn.BatchNorm2d(64)]
        block_5 = [torch.nn.ReLU()]
        block_6 = [Block1()]
        block_7 = [torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False), torch.nn.BatchNorm2d(32)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5, *block_6, *block_7)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
