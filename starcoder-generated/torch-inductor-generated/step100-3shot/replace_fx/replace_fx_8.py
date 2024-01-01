
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, x1, p=0.1)
        x3 = torch.rand_like(x1)
        x4 = x2 + x3
        x5 = torch.rand_like(x1)
        x6 = x2 + x3 + x5
        return x2, x4, x6
# Inputs to the model
x1 = torch.randn(1, 16, 16)
