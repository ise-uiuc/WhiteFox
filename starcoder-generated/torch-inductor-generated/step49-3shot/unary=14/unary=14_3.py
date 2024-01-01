
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose3 = torch.nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose3(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
