
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.modules.conv.ConvTranspose2d(3, 32, 3, 2)
        self.block2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.block1(x1)
        v2 = self.block2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
