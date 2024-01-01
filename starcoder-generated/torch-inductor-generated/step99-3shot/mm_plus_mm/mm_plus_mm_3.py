
class Model(torch.nn.Module):
    def __init__(self, p, q):
        self.w = torch.nn.Linear(p, q)
        self.k = torch.nn.Linear(p, q)
    def forward(self, t1, t2):
        out1 = self.w(t1)
        out2 = self.w(t2)
        out3 = self.k(t1)
        out4 = self.k(t2)
        out = out1 - out2 + out3 - out4
        return out

# Inputs to the model
t1 = torch.randn(5, 5)
t2 = torch.randn(5, 5)
