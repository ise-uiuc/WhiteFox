
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 4)
        self.linear_2 = torch.nn.Linear(2, 4)
        self.linear_3 = torch.nn.Linear(2, 2)
    def forward(self, x5):
        v0 = x5
        v1 = torch.nn.functional.linear(v0, self.linear_1.weight, self.linear_1.bias)
        v2 = v1.transpose(1, 2)
        v3 = torch.nn.functional.linear(v0, self.linear_2.weight, self.linear_2.bias)
        v4 = v3.transpose(1, 2)
        v5 = v2 + v4
        return torch.nn.functional.linear(v5, self.linear_3.weight, self.linear_3.bias)
# Inputs to the model
x5 = torch.randn(1, 2, 2)
