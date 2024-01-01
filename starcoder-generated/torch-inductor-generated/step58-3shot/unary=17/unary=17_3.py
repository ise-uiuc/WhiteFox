
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.ConvTranspose2d(3, 16, 56, stride=32, padding=8)

    def forward(self, x1):
        v1 = self.t1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
