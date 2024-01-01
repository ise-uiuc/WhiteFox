
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
