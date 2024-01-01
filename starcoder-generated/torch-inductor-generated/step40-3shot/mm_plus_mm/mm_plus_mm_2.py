
class Model(torch.nn.Module):
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = t1 + t1
        t3 = torch.mm(x, x)
        t4 = t3 + t3
        r = t2 + t4
        return r
# Inputs to the model
x = torch.randn(2, 2)
