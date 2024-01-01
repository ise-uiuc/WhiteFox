
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        b1 = x1.size(0)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.linear(v2)
        return v3.view(b1, -1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
