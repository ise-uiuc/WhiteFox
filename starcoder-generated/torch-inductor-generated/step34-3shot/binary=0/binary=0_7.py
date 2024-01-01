
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.randn(32))
        self.p2 = torch.nn.Parameter(torch.randn(32))
        self.p3 = torch.nn.Parameter(torch.randn(32))
        self.p4 = torch.nn.Parameter(torch.randn(32))
        self.w1 = torch.nn.Linear(1,32)
        self.w2 = torch.nn.Linear(3,16)
        self.w3 = torch.nn.Linear(16, 8)
    def forward(self, x1, x2):
        w1 = self.w1(x1)
        w2 = self.w2(x2).unsqueeze(dim=1)
        w3 = self.w3(w1 + self.p1) + self.p2 @ w2
        w4 = self.w1(x1) @ self.p4.unsqueeze(dim=0)
        return w3 + w4
# Inputs to the model
x1 = torch.randn(4, 1)
x2 = torch.randn(4, 3)
x = (x1, x2)

