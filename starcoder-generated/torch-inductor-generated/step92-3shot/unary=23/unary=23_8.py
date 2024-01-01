
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 5, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv_transpose(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 3, 4)
