
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(2, 2, 2, bias=True)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
