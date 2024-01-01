
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 2, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 4, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose2(self.conv_transpose(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
