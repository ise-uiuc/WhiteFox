
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        p1 = torch.nn.functional.dropout(x1)
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
