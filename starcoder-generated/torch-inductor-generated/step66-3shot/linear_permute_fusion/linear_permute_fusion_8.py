
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
    def forward(self, x2):
        v1 = x2
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = v1.flip(0)
        v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v6 = v3 + v5
        return v6.permute(1, 0, 2)
# Inputs to the model
x2 = torch.randn(3, 4)
