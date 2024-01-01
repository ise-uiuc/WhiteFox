
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose2d = torch.nn.ConvTranspose2d(5, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.convtranspose2d(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
