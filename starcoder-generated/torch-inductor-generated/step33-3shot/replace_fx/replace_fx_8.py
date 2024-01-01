
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        c = torch.rand_like(x1)
        a = x1 * c
        b = a * x2
        x4 = torch.nn.functional.dropout(b)
        x5 = torch.nn.functional.gelu(x4)
        return x5
# Inputs to the model
x3 = torch.randn(1, 3, 4)
x4 = torch.randn(1, 3, 4)
