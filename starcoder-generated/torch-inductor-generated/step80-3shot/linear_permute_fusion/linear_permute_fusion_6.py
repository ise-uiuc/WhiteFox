
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2):
        v = torch.cat((x1, x2), dim = 1)
        v1 = torch.nn.functional.linear(v, self.linear.weight, self.linear.bias)
        v2 = v1
        v2 = v2.permute(0, 2, 1)
        return v
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
