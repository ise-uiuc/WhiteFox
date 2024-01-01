
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 3, padding=0, bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 3, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 32, 32)
