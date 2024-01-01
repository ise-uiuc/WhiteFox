
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.tanh(v2)
        v4 = v3 + v2
        v4 = self.relu(v4)
        x2 = v4.permute(0, 2, 1)
        v3 = torch.matmul(x2, v4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
