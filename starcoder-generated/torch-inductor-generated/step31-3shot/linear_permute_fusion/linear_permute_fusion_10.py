
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        y = x1
        v1 = torch.nn.functional.linear(y, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        w = v2.size(1)
        h = v2.size(2)
        v3 = v2.contiguous().view(v2.size(0), w * h)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        w = v4.size(1)
        h = v4.size(2)
        v5 = v4.contiguous().view(v4.size(0), w * h)
        return v5
# Inputs to the model
x1 = torch.randn(3, 2, 2)
