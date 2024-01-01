
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 1, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2 + 0.0
        v2 = torch.max(v3, dim=-1)[0]
        v3 = v2 + 0.0
        v2 = v3 + 0.0
        v2 = v2 + 0.0
        v2 = v2 + 0.0
        v2 = v2 + 0.0
        v2 = v2 + 0.0
        v2 = v2 + 0.0
        v2 = v2 + 0.0
        v2 = v2.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = v2[0, :, 0]
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 2)
