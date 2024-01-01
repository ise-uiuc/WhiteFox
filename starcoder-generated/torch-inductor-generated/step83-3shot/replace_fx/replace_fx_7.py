
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = F.dropout(x2, p=0.5)
        x4 = F.dropout(x3, p=0.5)
        x5 = torch.rand_like(F.dropout(x4, p=0.5))
        return (x2, x5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
