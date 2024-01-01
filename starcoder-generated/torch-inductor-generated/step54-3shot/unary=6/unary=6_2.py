
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1a = x1 + 1
        x2a = x1a + 1
        x1 = x1 + 1
        x2 = x1 + 1
        x2b = x2 + 1
        x1b = x2b + 1
        x3 = F.hardtanh(torch.cat([x1a, x2a, x1b, x2b]), min_val=0, max_val=6)
        x3 = x3 / 6
        return x3
# Inputs to the model
x1 = torch.randn(2, 64, 256, 256)
