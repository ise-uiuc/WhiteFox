
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        c1 = torch.nn.functional.dropout(x1, p=0.3)
        c2 = torch.nn.functional.dropout(x1, p=0.5)
        c3 = torch.nn.Parameter(torch.rand(5, 4))
        c4 = torch.cat([c1, c2, c3], dim=2)
        c5 = torch.rand(4, 5)
        c6 = torch.nn.functional.linear(c5, c4) 
        return c6
# Inputs to the model
x1 = torch.randn(1, 2)
