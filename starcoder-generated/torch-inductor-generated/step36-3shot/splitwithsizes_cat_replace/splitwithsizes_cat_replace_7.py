
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 1, 1, bias=False),
            torch.nn.GroupNorm(2, 32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 2, 1, bias=False))
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 1, 1, bias=False),
            torch.nn.GroupNorm(2, 64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 1, 1, bias=False),
            torch.nn.GroupNorm(2, 32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 2, 1, bias=False))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 1, 1, bias=False),
            torch.nn.GroupNorm(2, 32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 1, bias=False))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, 1, 1, bias=True))
    def forward(self, inputs):
        x1, x2, x3 = self.layer0(inputs), self.layer1(x1), self.layer2(x2)
        y1 = self.layer3(x3)
        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        x3 = x3.view(x3.size()[0], -1)
        y1 = y1.view(y1.size()[0], -1)
        z1 = torch.cat((x1, x2, y1), dim=1)
        return z1 # Split the input tensor into several tensors along a given dimension
# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
