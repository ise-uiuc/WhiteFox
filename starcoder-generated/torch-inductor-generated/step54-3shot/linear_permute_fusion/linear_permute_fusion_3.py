
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
    def forward(self, x, y):
        v1 = torch.nn.functional.linear(x, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.linear(y, self.linear1.weight, self.linear1.bias)
        v3 = torch.stack([v1, v2])
        return v3.permute(1, 2, 3, 0)
# Inputs to the model
x = torch.randn(2, 2, 2)
y = torch.randn(1, 2, 2)
