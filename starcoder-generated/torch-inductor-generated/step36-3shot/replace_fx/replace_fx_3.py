
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x + x, p=0.5)
        x2 = torch.rand_like(x)
        x3 = x + x1 + x2
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
