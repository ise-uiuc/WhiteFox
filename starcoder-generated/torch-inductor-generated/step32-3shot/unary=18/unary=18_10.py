
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=(0, 3), groups=3, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 200, 300)
