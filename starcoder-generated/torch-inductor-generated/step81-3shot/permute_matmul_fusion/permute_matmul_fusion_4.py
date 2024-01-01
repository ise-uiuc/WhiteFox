
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        c1 = x.copy()
        v = c1.view(1, 2, 2, 2)
        v0 = v[:1,1:3,...]
        v1 = torch.bmm(v0, c1)
        return v1
# Inputs to the model
x = torch.randn(1, 2, 2)
