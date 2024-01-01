
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 3, 9, bias=True, padding=4, stride=1), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
    def forward(self, x1):
        y = self.block0(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
