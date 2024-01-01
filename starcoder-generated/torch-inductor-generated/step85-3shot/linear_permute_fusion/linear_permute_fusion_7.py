
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.linear2 = torch.nn.Linear(1, 2)
    def forward(self, x1):
        v3 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v1 = v4.permute(1, 0)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1)
