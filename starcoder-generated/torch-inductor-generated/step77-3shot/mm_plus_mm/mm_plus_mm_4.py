
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        y1 = torch.mm(x1, x2)
        y2 = torch.mm(y1, x3)
        y3 = torch.mm(y2, x4)
        y4 = torch.mm(y3, x5)
        y5 = torch.mm(y4, y0)
        return y5
# Inputs to the model
w = torch.randn(5, 5)
x = torch.randn(5, 5)
