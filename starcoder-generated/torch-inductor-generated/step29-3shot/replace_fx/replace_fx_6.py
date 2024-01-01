
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        g1 = torch.nn.functional.dropout(x1)
        g2 = torch.rand_like(x1)
        return g1
# Inputs to the model
x1 = torch.randn(1, 280, 10)
