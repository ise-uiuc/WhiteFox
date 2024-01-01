
class Model(torch.nn.Module):
    def __init__(self, d=0.3):
        super().__init__()
        self.d = d
    def forward(self, x):
        p1 = x ** (1./self.d)
        p2 = torch.nn.functional.dropout(x, p=self.d)
        p3 = p1 * p2 
        p4 = (1./(x + p3))
        return p4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
