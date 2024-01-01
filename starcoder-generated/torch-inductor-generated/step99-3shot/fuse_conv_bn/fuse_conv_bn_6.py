
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.conv1 = torch.nn.ConvTranspose2d(2, 5, 3, )
        self.bn1 = torch.nn.BatchNorm2d(5)
    def forward(self, x4):
        o4 = self.conv1(x4)
        x2 = self.bn1(o4)
        return x2
# Inputs to the model
x4 = torch.randn(1, 2, 4, 4)
