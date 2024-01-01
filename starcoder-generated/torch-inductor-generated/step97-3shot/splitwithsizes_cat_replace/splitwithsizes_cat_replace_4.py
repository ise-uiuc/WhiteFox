
v0 = [torch.randn(1, 3, 224, 224) for _ in range(3)]
v1 = [torch.randn(1, 3, 224, 224) for _ in range(3)]
v2 = [torch.randn(1, 3, 224, 224) for _ in range(3)]

class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 7, 1, 2, bias=True)
        self.conv2 = torch.nn.Conv2d(10, 3, 5, 1, 2, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        output1 = self.conv2(x)
        output2 = x.view(x.size()[0], x.size()[1] * x.size()[2] * x.size()[3])
        return (output1, output2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([Block() for _ in range(3)])
    def forward(self, v1):
        split_tensors = torch.split(torch.cat(v1, dim=1), [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
