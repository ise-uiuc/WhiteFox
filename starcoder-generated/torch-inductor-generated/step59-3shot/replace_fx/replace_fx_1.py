
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.rand_like(x2)
        x4 = F.dropout(x3)
        x5 = x2 + x4
        return F.dropout(x5, p=0.5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 3, 2)
