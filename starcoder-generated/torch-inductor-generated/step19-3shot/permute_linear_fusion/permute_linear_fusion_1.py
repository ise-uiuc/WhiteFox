
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ReLU = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = self.ReLU(v2)
        y = torch.matmul(v2, self.linear.bias)
        z = torch.nn.functional.relu(y)
        return (x2 + z) * x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
