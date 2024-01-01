
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v6 = x1
        v1 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 3, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
