
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp):
        r1 = x.view(1, 3*3)
        r2 = x.view(1, 3, 3)
        r3 = torch.flatten(r2)
        y = torch.cat((x, x))
        z = torch.stack([x, x])
        return (r1, r2, r3, y, z)
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
