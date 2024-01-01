
class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return [y1, y2]
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([Block()])
        self.classifier = torch.nn.ModuleList([Block()])
        self.other_features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), Block()])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
