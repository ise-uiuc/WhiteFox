
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v4 = x1
        v4 = v4.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
