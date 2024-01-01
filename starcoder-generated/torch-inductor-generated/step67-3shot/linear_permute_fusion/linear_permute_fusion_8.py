
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v2 = x1
        v1 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v3 = v1.permute(0, 2, 3, 1)
        v4 = v3.permute(0, 3, 1, 2)
        return v4.permute(0, 3, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
