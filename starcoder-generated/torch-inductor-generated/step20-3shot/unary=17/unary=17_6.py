
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 16, 3, padding=1, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 8, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)        
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
