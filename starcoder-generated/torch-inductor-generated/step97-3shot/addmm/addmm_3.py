
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.randn(3, 3, requires_grad=True)
        self.x2 = torch.randn(3, 3, requires_grad=True)
        self.w1 = torch.randn(5, 3)
    def forward(self, v3, v8):
        t1 = torch.mm(self.x1, self.x2)
        t2 = torch.mm(t1, self.w1)
        v1 = torch.mm(t2, self.x2)
        v2 = torch.mm(v1, v3)
        return v2 + v8
# Inputs to the model
v3 = torch.randn(3, 3, requires_grad=True)
v4 = torch.randn(5, 3, requires_grad=True)
v5 = torch.randn(3, 3)
v6 = torch.randn(3, 3)
v7 = torch.randn(3, 3)
v8 = torch.randn(3, 3, requires_grad=True)
