
class Model(torch.nn.Module):
    def __init__(self, p, d):
        super().__init__()
        self.p = p
        self.d = d
    def forward(self, x1, x2):
        x3 = torch.rand_like(x1)
        x4 = F.dropout(x2, p=self.d)
        if x3.sum() > self.p:
            x5 = x3 * x1
        else:
            x5 = x4 * x1
        return x5
p = 0.5
d = 0.5
# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(3, 2, 2)
