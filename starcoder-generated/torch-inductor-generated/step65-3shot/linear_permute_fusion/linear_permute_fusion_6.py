
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = torch.nn.Linear(2, 2)
        self.linear_1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear_0.weight, self.linear_0.bias)
        v2 = torch.nn.functional.linear(x1, self.linear_1.weight, self.linear_1.bias)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(3, 2, 2)
