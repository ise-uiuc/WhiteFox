
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 4, padding=2, stride=(2, 1), bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(32, 1, 4, padding=2, stride=(2, 1), bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
