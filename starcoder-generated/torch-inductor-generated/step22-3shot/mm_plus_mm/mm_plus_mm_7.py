
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(5, 5)
        self.w2 = torch.randn(5, 5)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, self.w1)
        v2 = torch.mm(x1, self.w2)
        v3 = torch.mm(self.w1, x2)
        v4 = torch.mm(self.w2, x2)
        v5 = v1+v2+v3+v4
        return v5+v5
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(4, 5)
