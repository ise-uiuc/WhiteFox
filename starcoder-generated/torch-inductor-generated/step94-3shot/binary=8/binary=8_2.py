
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn1.bias.requires_grad_(False)
        self.bn1.apply(torch.nn.init.uniform_.dirac_)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn2.bias.requires_grad_(False)
        self.bn2.apply(torch.nn.init.uniform_.dirac_)
    def forward(self, x1, x2):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.bn2(self.conv2(x2))
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
