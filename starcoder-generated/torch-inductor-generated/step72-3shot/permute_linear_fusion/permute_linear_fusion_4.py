
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.hardtanh1 = torch.nn.Hardtanh(min_val=-6.0, max_val=1.0)
        self.hardtanh2 = torch.nn.Hardtanh(min_val=-1.0, max_val=6.0)
        self.gelu = torch.nn.GELU()
    def forward(self, x2):
        v1 = x2.permute(0, 2, 1).permute(0, 2, 1).permute(0, 2, 1).permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight * -0.1 + 0.0, self.linear.bias)
        v3 = self.gelu(v2)
        v4 = self.hardtanh1(v3)
        return self.hardtanh2(v4)
# Inputs to the model
x2 = torch.randn(1, 2, 2)
