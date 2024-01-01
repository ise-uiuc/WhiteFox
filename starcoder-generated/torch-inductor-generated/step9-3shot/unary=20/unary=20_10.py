
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose3d(1, 6043, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.tconv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 9, 19, 18)
