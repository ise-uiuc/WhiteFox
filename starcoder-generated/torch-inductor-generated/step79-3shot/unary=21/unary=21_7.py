
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(7, stride=16)
    def forward(self, x):
        y = self.avg_pool(x)
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.randn(1, 256, 14, 14)
