
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1025)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = v2.transpose(1, 2)
        x3 = x2
        x4 = x2.transpose(2, 3)
        v4 = torch.clamp(x4, 0, 1)
        v5 = x2 + v4
        x4 = x2.transpose(1, 2)
        v5 = x4.transpose(0, 2).transpose(2, 3)
        return x2
# Inputs to the model
x1 = torch.randn(1, 4, 2)
