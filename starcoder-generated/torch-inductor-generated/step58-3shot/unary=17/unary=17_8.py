
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transposeconv = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.transposeconv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
