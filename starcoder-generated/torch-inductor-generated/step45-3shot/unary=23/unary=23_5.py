
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 3, 5, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 3, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.tanh(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
