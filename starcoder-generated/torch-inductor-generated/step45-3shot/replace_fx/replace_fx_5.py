
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.0)
        x3 = torch.rand_like(x1)
        x4 = x3 - 0
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
