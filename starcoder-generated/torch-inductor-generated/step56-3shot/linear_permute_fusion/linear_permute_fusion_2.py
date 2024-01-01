
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.linear_1 = torch.nn.Linear(2, 2)
    def forward(self, x3):
        v0 = torch.reshape(x3, (1, 2, 2))
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = self.linear_1.weight
        v4 = torch.nn.functional.linear(v2, v3, self.linear.bias)
        return v4
# Inputs to the model
x3 = torch.randn(1, 2, 2)
