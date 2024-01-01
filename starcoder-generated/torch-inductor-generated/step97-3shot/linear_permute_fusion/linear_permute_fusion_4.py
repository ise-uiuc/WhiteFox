
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x3):
        v0 = x3
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.flip(-1)
        return v3.transpose(-1, -2).contiguous()
# Inputs to the model
x3 = torch.randn(1, 2, 2)
