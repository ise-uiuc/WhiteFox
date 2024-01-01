
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = torch.mm(x1, x1)
        v = torch.cat(v, 1)
        v = v.view(((1,2,3), (4,5,6)))
        return v
# Inputs to the model
x1 = torch.randn(6, 6)
