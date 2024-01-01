
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.gelu = torch.nn.GELU()
        self.gelu2 = torch.nn.GELU()
        self.gelu3 = torch.nn.GELU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.gelu(v2)
        v4 = self.gelu2(v3)
        v5 = self.gelu3(v3)
        return torch.abs(torch.abs(v2) - torch.abs(v3) - torch.abs(v4) - torch.abs(v5))
# Inputs to the model
x1 = torch.randn(1, 2, 3)
