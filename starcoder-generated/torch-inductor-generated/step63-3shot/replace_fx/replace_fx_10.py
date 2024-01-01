
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = torch.rand_like(x1)
        x4 = torch.mul(x2, x3)
        x5 = torch.sum(x4)
        x6 = F.dropout(x5)
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
