
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(2, 2)
        self.w2 = torch.randn(2, 2)
        self.w3 = torch.randn(2, 2)
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, self.w1)
        v2 = torch.mm(x2, self.w2)
        v3 = torch.mm(x3, self.w3)
        v4 = torch.mm(x4, self.w2)
        return v1 + v2 + v3 + v4
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)
