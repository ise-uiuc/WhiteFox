
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.ConvTranspose2d(3, 12, 3, bias=False, padding=1, stride=2)
        self.block1 = torch.nn.ReLU(inplace=False)
        self.block2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.block0(x1)
        v2 = self.block1(v1)
        v3 = self.block2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
