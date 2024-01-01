
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 10, 3, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 4, kernel_size=(7, 14), stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 32, 16)
