
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.6)
        a2 = torch.randn(a1.size())
        a3 = torch.rand_randint(0, 1, a1.size(), dtype=torch.double)
        a4 = torch.rand_like(a1)
        # a5 = torch.randn(3, 4)
        return torch.nn.functional.dropout(a1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
