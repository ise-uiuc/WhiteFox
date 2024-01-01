
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 2, 2, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 3, 4, stride=3)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(3, 2, 2, stride=2)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(2, 1, 2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv_transpose3(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv_transpose4(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
