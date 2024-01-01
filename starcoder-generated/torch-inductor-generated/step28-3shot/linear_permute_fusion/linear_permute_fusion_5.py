
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
    def forward(self, x1):
        v2 = self.linear(x1)
        v1 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 3)
