
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = v3.exp()
        v5 = v3.log()
        x2 = v4 / v5
        v6 = torch.rand_like(x2)
        y = x2.squeeze() + v6
        w = v2 + y
        return w
# Inputs to the model
x1 = torch.randn(1, 2, 2)
