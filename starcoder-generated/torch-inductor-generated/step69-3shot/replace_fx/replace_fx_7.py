
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1, dtype=torch.float64)
        x3 = F.dropout(x2, p=0.5)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
