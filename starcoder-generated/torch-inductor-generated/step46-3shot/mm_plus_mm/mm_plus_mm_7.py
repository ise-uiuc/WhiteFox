
class Model(torch.nn.Module):
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.mm(x, x)
        v = torch.mm(x, x)
        v = v + torch.mm(t1, t2)
        return v
# Inputs to the model
x = torch.randn(4, 4)
