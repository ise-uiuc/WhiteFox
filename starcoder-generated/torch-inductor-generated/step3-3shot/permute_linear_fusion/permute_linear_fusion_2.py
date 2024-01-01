
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(1, 3, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = self.conv(v3)
        v5 = v4.permute(0, 2, 1)
        v6 = self.relu(v5)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
