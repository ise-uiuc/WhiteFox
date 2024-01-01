
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(3, 32, stride=2, kernel_size=3, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(32, 3, stride=1, kernel_size=3, padding=1)
    def forward(self,x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20, 20)
