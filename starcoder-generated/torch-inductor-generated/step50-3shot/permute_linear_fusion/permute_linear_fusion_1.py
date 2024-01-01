
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.randn_like(v2)
        x3 = x2.to(torch.float32)
        v5 = v2.to(dtype=torch.float32)
        v3 = torch.sigmoid(v2)
        v4 = v3 + x2 + x3 + v5
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
