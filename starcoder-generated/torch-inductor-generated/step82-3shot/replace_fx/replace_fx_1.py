
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 + 1.0
        x2 = self.randlike()
        return x2
    def randlike(self):
        x3 = torch.rand_like(x1)
        return self.dropout(x3)
    def dropout(self, x1):
        return F.dropout(x1, p=0.5)       
# Inputs to the model
x1 = torch.randn(1, 2, 2)
