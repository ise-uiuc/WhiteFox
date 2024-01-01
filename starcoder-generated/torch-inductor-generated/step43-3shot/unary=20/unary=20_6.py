
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.t(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
