
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 10)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2_1 = v2 + v1
        x2 = torch.nn.functional.tanh(v2_1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 4, 8)
