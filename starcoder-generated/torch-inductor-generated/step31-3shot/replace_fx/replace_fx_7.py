
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = F.dropout(x1)
        x3 = F.dropout(x1)
        x4 = torch.rand_like(x2)
        x5 = torch.rand_like(x3)
        x6 = F.dropout(x5)
        x7 = F.dropout(x5)
        return x6, x7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
