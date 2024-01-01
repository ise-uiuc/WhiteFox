
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 4)
        self.linear_2 = torch.nn.Linear(2, 2)
    def forward(self, x3):
        v0 = x3
        v1 = torch.nn.functional.linear(v0, self.linear_1.weight, self.linear_1.bias)
        v2 = v1.permute(0, 2, 1)
        return torch.nn.functional.linear(v2, self.linear_2.weight, self.linear_2.bias)
# Inputs to the model
x3 = torch.randn(1, 2, 2)
