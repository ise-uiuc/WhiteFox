
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, weight1, weight2, weight3, weight4, weight5):
        out = torch.mm(x1, weight1)
        out = torch.addmm(x2, weight2, out)
        out = torch.add(out, weight3)
        out = torch.mm(x1, weight4)
        out = torch.add(out, weight5)
        return out
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3)
weight1 = torch.randn(3, 3)
weight2 = torch.randn(3, 3)
weight3 = torch.randn(3, 3)
weight4 = torch.randn(3, 3)
weight5 = torch.randn(3, 3)
