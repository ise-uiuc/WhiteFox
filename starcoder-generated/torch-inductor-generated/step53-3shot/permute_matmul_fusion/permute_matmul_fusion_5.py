
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        w1 = x1.permute(1, 2, 0)
        y1 = torch.bmm(x2, w1)
        w2 = x2.permute(1, 2, 0)
        y2 = torch.bmm(w2, x1)
        y3 = y1.permute(1, 2, 0)
        return torch.matmul(x2, y3)
# Inputs to the model
x1 = torch.randn(3, 2, 2)
x2 = torch.randn(3, 2, 2)
