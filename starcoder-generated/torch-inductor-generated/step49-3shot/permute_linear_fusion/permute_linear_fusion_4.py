
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv1 = torch.nn.Conv2d(2, 2, 2, stride=1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
