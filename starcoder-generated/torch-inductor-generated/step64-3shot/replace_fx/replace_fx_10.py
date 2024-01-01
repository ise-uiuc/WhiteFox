
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.rand_like(x)
        x2 = F.dropout(x, p=0.5)
        x3 = F.dropout(x1, p=0.5)
        x4 = F.dropout(x2, p=0.5)
        x5 = F.dropout(x3, p=0.5)
        x6 = F.dropout(x4, p=0.5)
        x7 = F.dropout(x5, p=0.5)
        x8 = F.dropout(x6, p=0.5)
        return x8
# Inputs to the model
x = torch.randn(1, 2, 2, 2)
