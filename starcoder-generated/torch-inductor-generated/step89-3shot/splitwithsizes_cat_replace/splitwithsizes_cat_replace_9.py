
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
    def forward(self, inputs):
        split_tensors = torch.split(inputs, [1, 1, 1], dim=1)
        concatenated_tensors = torch.cat([split_tensors[i] for i in range(len(split_tensors))], dim=1)
        return concatenated_tensors
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(*(Block() for i in range(4)))
        self.conv1 = torch.nn.Conv2d(32, 32, 3)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
