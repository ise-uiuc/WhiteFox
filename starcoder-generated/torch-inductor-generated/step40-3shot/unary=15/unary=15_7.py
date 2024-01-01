
class Model(torch.nn.Module):
    def __init__(self, x: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, x, (1))
        self.conv2 = torch.nn.Conv1d(1, x, (1))
    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        v1 = self.conv1(x1)
        v2 = torch.clamp(v1, min=1000, max=10000)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3).reshape([1, x])
        v5 = torch.softmax(v4, -1)
        return v5
# Inputs to the model
x = 10
x1 = torch.randn(1, 1, 100)
