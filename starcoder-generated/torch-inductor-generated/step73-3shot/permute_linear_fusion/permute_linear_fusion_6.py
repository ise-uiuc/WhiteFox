
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(500, 50)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        x2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        z = torch.nn.functional.sigmoid(x2)
        v2 = z.permute(0, 2, 1)
        x3 = torch.nn.functional.linear(v2, self.linear.bias)
        m1 = x2 + x3
        m2 = m1.permute(0, 2, 1)
        y = torch.nn.functional.linear(m2, self.linear.bias)
        return m2
# Inputs to the model
x1 = torch.randn(1, 500, 2)
