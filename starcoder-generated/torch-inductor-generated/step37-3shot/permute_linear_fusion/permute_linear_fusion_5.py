
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_2 = torch.nn.Linear(3, 1)
    def forward(self, x, y):
        v1 = x.permute(0, 2, 1)
        v1 = v1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear_2.weight, self.linear_2.bias)
        return torch.sum(v2)
# Inputs to the model
x = torch.randn(1, 2, 2)
y = torch.randn(1, 1, 3)
