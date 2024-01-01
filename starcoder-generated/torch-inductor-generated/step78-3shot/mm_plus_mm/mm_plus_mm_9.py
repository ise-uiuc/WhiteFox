
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, X):
        t1 = torch.mm(X, X)
        t2 = t1 + 1
        t3 = torch.mm(10 * t2, X)
        t4 = 5 + t3
        t5 = torch.mm(10 * torch.mm(X, t4), t4)
        return t5
# Inputs to the model
X = torch.randn(3, 3)
