
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.cat([x1, x2], 0)
        t2 = torch.add(t1, x2)
        t3 = t1 * torch.sigmoid(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 1, 2, 2)
