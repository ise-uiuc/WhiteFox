
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t = F.dropout(x1, p=0.5)
        y1 = torch.nn.functional.gelu(t)
        w = torch.rand_like(x1, dtype=torch.float)
        y2 = y1 + w
        y3 = F.dropout(y2)
        return y3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
