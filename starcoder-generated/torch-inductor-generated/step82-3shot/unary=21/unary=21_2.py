
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)
        self.linear = torch.nn.Conv2d(2, 10, 32)
    def forward(self, x):
        v1 = self.pool(x)
        v2 = torch.tanh(v1)
        v3 = self.linear(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 2, 32, 32)
