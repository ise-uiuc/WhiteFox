
class Model(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
        self.weight = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.bias = bias
    def forward(self, x1):
        v1 = F.conv2d(x1, self.weight, self.bias)
        v2 = v1 + 3
        v3 = nn.functional.relu(v2, 0, 6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
