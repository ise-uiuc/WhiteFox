
class Module7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 2, 2, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 3, 2, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(3, 1, 2, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv_transpose3(v4)
        return torch.tanh(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
