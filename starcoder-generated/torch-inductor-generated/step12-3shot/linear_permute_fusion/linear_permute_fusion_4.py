
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.tanh(v1)
        v3 = v2.permute(0, 2, 1)
        return v2 * v3
# Inputs to the model
x = torch.randn(2, 4, 4)
