
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.expand(2, 2, 2)
        v3 = v2.permute(0, 1, 2).contiguous()
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 3)
