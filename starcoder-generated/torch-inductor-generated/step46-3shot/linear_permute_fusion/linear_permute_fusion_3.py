
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v4 = x1
        v2 = v4.view(2, -1)
        v3 = v2.permute(1, 0)
        v1 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        return v1
# Inputs to the model
x1 = torch.randn(3, 4)
