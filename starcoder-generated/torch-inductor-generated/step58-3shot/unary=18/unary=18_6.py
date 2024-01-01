
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=6, padding=2, use_bias=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
