
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hardswish = torch.nn.Hardswish()
        self.hardtanh = torch.nn.Hardtanh(-2, 3)
    def forward(self, x1):
        v1 = x1.permute(...)
        v2 = self.hardtanh(v1)
        v3 = self.hardswish(v2)
        v4 = v1 + v2
        v5 = v2 + v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, 2)
