
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randint(2, (1,), dtype=torch.int32)
        x3 = torch.randint(0, 3, (1,), dtype=torch.int32)
        x4 = F.dropout(x1, p=float(x2), training=bool(x3))
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
