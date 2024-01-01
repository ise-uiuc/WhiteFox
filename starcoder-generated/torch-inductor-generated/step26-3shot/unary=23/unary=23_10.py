
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 23, 2, stride=12, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 19, 29)
