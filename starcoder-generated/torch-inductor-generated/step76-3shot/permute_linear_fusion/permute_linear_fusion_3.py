
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(28, 128)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = x1 + 123.719
        x1b = self.linear1(v3) * 1.37
        v3b = v3 + 4.54
        v4 = torch.nn.functional.relu(x1b)
        x2 = torch.nn.functional.relu(x1)
        x3 = x2 + 1
        v4 = v4.permute(0, 2, 1)
        v5 = torch.sigmoid(v4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 28, 2)
