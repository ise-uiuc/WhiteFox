
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = v1.permute(1, 0)
        v2 = v3.permute(1, 0)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 2, 2)
