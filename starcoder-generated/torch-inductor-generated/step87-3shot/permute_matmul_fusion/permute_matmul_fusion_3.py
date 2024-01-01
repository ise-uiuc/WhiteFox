
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(1, 0, 2)
        t2 = torch.bmm(t1, x2)
        t3 = t2.permute(1, 0, 2)
        return t3
# Inputs to the model
x1 = torch.randn(2, 1, 2)
x2 = torch.randn(1, 2, 2)
