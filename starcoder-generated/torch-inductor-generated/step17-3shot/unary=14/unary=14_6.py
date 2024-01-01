
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 9, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
