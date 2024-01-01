
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.rand(4, 10)
        bias = torch.rand(4)
        self.conv1 = torch.nn.Conv1d(3, 4, 1)
        self.conv1.weight = torch.nn.Parameter(weight)
        self.conv1.bias = torch.nn.Parameter(bias)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * v1
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224)
