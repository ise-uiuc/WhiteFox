
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Block1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block1()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        block_0 = self.block.block(concatenated_tensor)
        return (concatenated_tensor, torch.split(block_0, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
