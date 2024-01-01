
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.flip(0)
        v4 = v3.transpose(0, 1)
        return v4
# Inputs to the model
x1 = torch.randn(80, 960, 2)
