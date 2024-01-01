
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t2 = x2.permute(0, 3, 1, 2)
        t3 = x1.permute(0, 3, 1, 2)
        t5 = torch.bmm(t3, t2)
        v8 = t2.permute(0, 1, 3, 2)
        v9 = torch.bmm(t2, v8)
        t4 = x1.permute(0, 3, 1, 2)
        v6 = torch.bmm(t4, v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
x2 = torch.randn(1, 3, 2, 2)
