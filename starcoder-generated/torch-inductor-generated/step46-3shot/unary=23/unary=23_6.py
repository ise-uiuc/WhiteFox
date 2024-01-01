
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
