
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        m = v2.max(0)
        y = v2.mul(m.values)
        for i in range(2):
            y1 = y.clone()
            y1[0][i] = 12.34
        y1.fill_(11.2)
        return y1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
