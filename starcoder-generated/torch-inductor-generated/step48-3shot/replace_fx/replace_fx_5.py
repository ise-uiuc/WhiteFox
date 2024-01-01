
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.add
        self.l2 = torch.mul
        self.l3 = torch.sub
    def forward(self, x1):
        t = F.dropout(x1, p=0.5)
        x2 = torch.sin(t)
        x3 = self.l1(x2, x2)
        x4 = torch.abs(x3)
        x5 = self.l3(x2, x2)
        return x5
# Inputs to the model
x1 = torch.randn(5, 5)
