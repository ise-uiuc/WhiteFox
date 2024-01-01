
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.hardtanh1 = torch.nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.layernorm = torch.nn.LayerNorm(2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.hardtanh1(v2)
        return self.layernorm(v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
