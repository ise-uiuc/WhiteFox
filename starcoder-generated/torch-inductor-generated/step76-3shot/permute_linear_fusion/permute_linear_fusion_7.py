
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 512)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(torch.nn.functional.tanh(x1 + 0.37), self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(torch.flatten(x1, 1), self.linear.weight / 2, self.linear.bias)
        v3 = torch.matmul(v2, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2048, 2)
