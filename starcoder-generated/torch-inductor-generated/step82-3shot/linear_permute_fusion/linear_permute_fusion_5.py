
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1[0] - v1[1]
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 2, 3)
