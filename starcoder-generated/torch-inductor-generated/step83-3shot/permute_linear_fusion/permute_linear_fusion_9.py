
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = x1.permute(2, 0, 1)
        v2 = self.tanh(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias))
        v3 = torch.min(v2, dim=-1)[1]
        v4 = torch.tanh(v3) + v2
        v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
