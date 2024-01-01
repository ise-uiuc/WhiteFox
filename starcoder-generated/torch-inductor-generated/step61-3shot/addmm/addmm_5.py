
class Model(torch.nn.Module):
    def forward(self, t1, t2, x1, x2):
        v1 = torch.mm(x1, x1)
        out = v1 + v1
        t1 = t1 + t2
        v1 = torch.mm(t2, t2)
        v1 = torch.mm(v1, t1)
        out = v1 + v1
        v1 = torch.mm(x1, x2) + out
        return v1
# Inputs to the model
t1 = torch.randn(3, 3, requires_grad=True)
t2 = torch.randn(3, 3)
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
mm1 = torch.randn(3, 3, requires_grad=True)
mm2 = torch.randn(3, 3, requires_grad=True)
