
class Model(torch.nn.Module):
    def forward(self, x):
        t0 = torch.mm(x, x)
        t1 = torch.mm(x, x)
        t2 = t1[:1,:]
        t3 = t2*torch.tensor(3.0)
        t4 = torch.mm(t0, t0)
        out = torch.mm(t0, x)
        out = t2 + t0 + out
        return out
# Inputs to the model
x = torch.randn(1, 10)
