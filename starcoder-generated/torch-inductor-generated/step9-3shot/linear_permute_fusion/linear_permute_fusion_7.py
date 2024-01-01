
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v3 = x1
        v2 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v1 = v2.permute(0, 3, 1, 2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
