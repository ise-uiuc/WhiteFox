
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1.permute(0, 2, 1), self.linear.weight, self.linear.bias)
        v2 = x1 + x1
        v3 = v2.reshape(v2.size()[0], v2.size()[1], 1, 1)
        v4 = v3 > v2
        v5 = v4.sum()
        return v1 + v2 * v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
