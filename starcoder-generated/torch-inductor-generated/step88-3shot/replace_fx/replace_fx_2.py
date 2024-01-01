
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 + 1.0
        x2 = F.dropout(x1)
        x3 = F.dropout(x2)
        x4 = F.dropout(x3)
        x5 = F.dropout(x4)
        x6 = x5.norm()
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
