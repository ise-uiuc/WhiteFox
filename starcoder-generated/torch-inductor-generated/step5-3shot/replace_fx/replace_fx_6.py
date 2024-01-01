
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1)
        a2 = torch.rand_like(x1)
        return torch.abs(a2)
# Inputs to the model
x1 = torch.randn(1, 2)
