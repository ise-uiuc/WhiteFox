
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(45, 4, 3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 45, 12, 12)
