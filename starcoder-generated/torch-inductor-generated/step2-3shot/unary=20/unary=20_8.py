
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, padding=2)
    def forward(self, x):
        v1 = self.tconv(x)
        return v1
# Inputs to the model
x = torch.randn(3, 3, 8, 8)
