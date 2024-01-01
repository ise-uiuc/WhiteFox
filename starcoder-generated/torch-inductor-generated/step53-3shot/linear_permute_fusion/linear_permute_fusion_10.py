
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 1)
        self.linear_2 = torch.nn.Linear(2, 1)
    def forward(self, x3, x4, x5):
        v2 = x4.flatten(0, 1)
        v3 = x5.flatten(0, 1)
        v4 = torch.cat((v2, v3), 0)
        v5 = torch.stack((v2, v3), 0)
        v6 = v4.transpose(0, 1)
        return v4, v5, v6
        v1 = x3
        v2 = torch.nn.functional.linear(v1, self.linear_1.weight, self.linear_1.bias)
        v3 = torch.nn.functional.linear(v1, self.linear_2.weight, self.linear_2.bias)
        v4 = (v2 == v3)
        v5 = v2 > v3
        v6 = v4 | v5
        return v4, v5, v6
# Inputs to the model
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(2, 1)
x5 = torch.randn(1, 2, 1)
