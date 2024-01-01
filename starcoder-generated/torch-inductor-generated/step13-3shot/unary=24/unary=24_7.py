
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.nn.functional.conv1d(x, weight=torch.zeros([8, 1, 10], dtype=torch.float), bias=torch.zeros(8, dtype=torch.float))
        v2 = v1 > 0
        v3 = v1 * 10
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 128)
