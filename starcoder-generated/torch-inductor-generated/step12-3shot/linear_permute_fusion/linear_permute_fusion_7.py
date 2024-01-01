
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2):
        v1 = x1.reshape((1, -1))
        v4 = x2.reshape((1, -1))
        v5 = v1 * self.linear.weight
        v6 = torch.addmm(v5, v4, x1, self.linear.bias)
        return v6.reshape((1,) + x1.size())
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
