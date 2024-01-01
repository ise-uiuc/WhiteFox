
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.0)
        a2 = torch.rand_like(x1)
        a3 = torch.randn(1)
        a4 = a2 - a3
        return torch.nn.functional.dropout(a1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
