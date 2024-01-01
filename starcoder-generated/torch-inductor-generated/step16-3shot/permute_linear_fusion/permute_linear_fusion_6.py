
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.tanh(v1)
        v4 = v3 * x2
        v4 = v4 + v2
        v5 = x2.permute(1, 0)
        v5 = torch.nn.functional.linear(v5, self.linear.weight, self.linear.bias)
        v6 = torch.nn.functional.tanh(v1)
        v7 = v6.transpose(1, 1)
        v8 = x2.permute(1, 0)
        v9 = v8.transpose(1, 1)
        v10 = torch.nn.functional.linear(v7, v9, None)
        v3 = torch.matmul(x2, v10)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
