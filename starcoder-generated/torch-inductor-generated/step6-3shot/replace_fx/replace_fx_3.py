
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a1 = torch.randn(2, 2)
        a2 = torch.nn.functional.sigmoid(x1)
        c2 = torch.nn.functional.relu(a2)
        g5 = torch.rand_like(a1)
        h3 = torch.nn.functional.softmax(g5, dim=0)
        f3 = torch.nn.functional.relu(h3)
        f2 = torch.add(x2, f3)
        f1 = torch.nn.functional.softmax(f2)
        f0 = torch.nn.functional.sigmoid(a2)
        return f0
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(3, 2, 2)
