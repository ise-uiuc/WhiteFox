
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        max1 = torch.max
        v3 = max(v2, -1, True)
        v4 = v3.unsqueeze(0)
        return v4
# Inputs to the model
x1 = torch.randn(3, 2, 2)
