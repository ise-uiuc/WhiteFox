
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.view(1, 2, 2)
        v3 = torch.zeros_like(v2)
        v4 = torch.zeros_like(v2)
        v5 = torch.cat((v3, v2, v4), 2)
        v6 = torch.stack((v5, v2), 0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
