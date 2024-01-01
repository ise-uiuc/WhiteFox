
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
        x4 = torch.rand_like(x1)
        x5 = torch.rand_like(x1)
        x6 = torch.rand_like(x1)
        x7 = torch.rand_like(x1)
        x = F.dropout(x1, p=0.5)
        x = F.dropout(x, p=0.5)
        return x2 + x3 + x4 + x5 + x6 + x7
# Inputs to the model
x1 = torch.randn(1, 2, 3)
