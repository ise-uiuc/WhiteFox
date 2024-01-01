
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.2)
        x3 = torch.rand_like(x2)
        x4 = self.randlike_a(x2)
        x5 = F.dropout(x4, p=0.5)
        return x5
    def randlike_a(self, x1):
        x3 = torch.rand_like(x1, dtype=torch.float)
        x4 = torch.rand_like(x3)
        return torch.rand_like(x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
