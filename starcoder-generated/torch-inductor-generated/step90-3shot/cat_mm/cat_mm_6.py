
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x2, x1)
        t = [v]
        for _ in range(5):
            t2 = torch.mm(x2, x1)
            t.append(t2)
        return torch.cat(t,1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(1, 3)
