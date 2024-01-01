
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.norm = MyModule()
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x1):
        x3 = self.conv1(x1)
        x4 = self.norm(x1)
        x5 = self.conv2(x4)
        l = torch.cat([x3, x5], dim=0)
        return l
# Inputs to the model
x1 = torch.randn(3, 1, 2, 2)
