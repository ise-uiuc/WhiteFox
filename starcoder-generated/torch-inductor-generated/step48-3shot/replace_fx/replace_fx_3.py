
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = torch.rand_like(x1, dtype=torch.float)
        x4 = F.dropout(x1, p=0.8490)
        x6 = F.dropout(x4)
        x7 = torch.nn.functional.gelu(x3)
        x8 = torch.nn.functional.gelu(x6)
        x9 = torch.nn.functional.gelu(x8)
        x10 = torch.nn.functional.gelu(x9)
        x10 = (x7)
        return x10
# Inputs to the model
x1 = torch.randn(1, 2, 2)
