
# Please add your comments on the model!
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        s1 = torch.mm(x1, x2)
        y1 = torch.mm(x1, x2)
        y2 = torch.mm(x1, x2)
        s2 = torch.mm(y1, y2)
        return torch.cat([s1, s2], 1)
x1 = torch.randn(2, 2)
x2 = torch.randn(1, 2)
