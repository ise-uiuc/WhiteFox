
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.maxpool2d(x, kernel_size=1, stride=1)
        v2 = x + v1
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
