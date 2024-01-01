
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ReLU = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.ReLU(v2)
        x2 = torch.matmul(v2, self.linear.bias)
        y = torch.nn.functional.relu(x2)
        z = (v3 + y) / 2
        w = x1 + v2
        return torch.nn.functional.relu(w + z)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
