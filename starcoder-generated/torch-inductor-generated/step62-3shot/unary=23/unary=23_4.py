
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = torch.nn.AvgPool2d(kernel_size=5, stride=3, padding=3, ceil_mode=True)
    def forward(self, x1):
        v1 = self.pooling(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
