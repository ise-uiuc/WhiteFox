
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(4, 1, 5, stride=(4, 2), padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64) 
