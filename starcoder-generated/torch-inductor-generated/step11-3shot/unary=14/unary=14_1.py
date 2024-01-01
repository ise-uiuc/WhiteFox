
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.ConvTranspose1d(17, 17, 3, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.c1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 17, 94)
