
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.rand_like(x1)
        x4 = torch.rand_like(x2)
        x5 = F.dropout(x1, p=0.5)
        x6 = F.dropout(x1, p=0.5)
        x7 = x7 + x6
        x8 = torch.rand_like(x3)
        return x6, x7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
