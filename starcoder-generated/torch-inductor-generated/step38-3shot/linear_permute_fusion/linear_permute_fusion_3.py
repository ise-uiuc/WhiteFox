
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear_v3 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(x1, self.linear_v3.weight, self.linear_v3.bias)
        v4 = v3.permute(0, 2, 1)
        return (v1, v2, v3, v4)
# Inputs to the model
x1 = torch.randn(3, 2, 2)
