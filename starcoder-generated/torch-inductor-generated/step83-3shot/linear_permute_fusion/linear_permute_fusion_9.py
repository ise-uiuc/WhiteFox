
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x10):
        v10 = x10
        v12 = torch.nn.functional.linear(v10, self.linear.weight, self.linear.bias)
        v11 = v12.permute(0, 2, 1)
        v9 = v12
        v12 = v9.permute(0, 2, 1)
        return v11 + v12
# Inputs to the model
x10 = torch.randn(1, 2, 2)
