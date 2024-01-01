
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x3):
        v0 = x3
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 3, 1)
        v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 3, 2, 1)
        return v4
# Inputs to the model
x3 = torch.randn(1, 2, 2)
