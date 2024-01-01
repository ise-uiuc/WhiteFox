
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=True)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.hardtanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
