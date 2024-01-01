
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, 8, stride=(4, 1), padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 4, 2, stride=1, padding=(1, 2))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(16, 16, 2, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = self.conv_transpose3(v3)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
