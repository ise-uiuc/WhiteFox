
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v0 = torch.tanh(x1)
        v1 = torch.max_pool2d(v0, 2)
        v2 = torch.conv_transpose2d(v1, 4, 6, (4, 4))
        v3 = torch.max_pool2d(v2, 2)
        v4 = torch.conv2d(v3, 1, (3, 3))
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 6, 6)
