
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.tanh(v2)
        x2 = torch.nn.functional.threshold(v2, -0.8, 0.8, False)
        v4 = x2 * x2
        v3 = torch.nn.functional.linear(v1, self.linear.weight * 2, self.linear.bias)
        v3 = v3 + x2
        return v2 + v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
