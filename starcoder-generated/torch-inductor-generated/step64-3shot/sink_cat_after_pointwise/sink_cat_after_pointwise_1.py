
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        a = a.clone().transpose(1, 2)
        b = b.clone()
        c = torch.cat((a, b), dim=0)
        d = c.squeeze()
        e = torch.cat((c, d), dim=0)
        e += b
        f = e.mean()
        return f
# Inputs for the model
a = torch.randn(2, 1, 3)
b = torch.randn(2, 3)
