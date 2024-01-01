
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(64, 32, 3), torch.nn.ReLU6())
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(32, 16, 3, bias=True), torch.nn.BatchNorm2d(16), torch.nn.ReLU6())
    def forward(self, x1):
        s1 = self.layer2(x1)
        s2 = self.layer(s1)
        return s2
# Inputs to the model
x1 = torch.randn(4, 64, 4, 4)
