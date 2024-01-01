
class Model(torch.nn.Module):
    def forward(self, a, b, c, d, e, f):
       t1 = torch.mm(a, b)
       t2 = torch.nn.functional.conv2d(c, d)
       t3 = torch.mm(e, f)
       return t1 + t2 + t3
# Inputs to the model
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.rand(2, 2)
d = torch.rand(2, 2)
e = torch.rand(2, 2)
f = torch.rand(2, 2)
