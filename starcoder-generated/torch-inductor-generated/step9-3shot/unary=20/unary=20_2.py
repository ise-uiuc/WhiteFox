
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1 = torch.nn.ConvTranspose2d(3, 4, 9, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.convtranspose1(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
