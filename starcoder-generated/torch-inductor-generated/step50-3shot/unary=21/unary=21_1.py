
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool2d = torch.nn.AvgPool2d((2, 2), padding=0)
    def forward(self, x):
        v1 = self.avg_pool2d(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 49, 96)
