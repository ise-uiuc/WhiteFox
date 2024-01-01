
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 7, stride=1, padding=2)
        self.conv2d = torch.nn.Conv2d(32, 480, 3, stride=1, padding=1, bias=False)
    def forward(self, x0):
        v0 = self.conv_transpose(x0)
        v1 = self.conv2d(v0)
        return torch.clamp(v1, min=0)
# Inputs to the model
x0 = torch.randn(1, 3, 512, 512)
