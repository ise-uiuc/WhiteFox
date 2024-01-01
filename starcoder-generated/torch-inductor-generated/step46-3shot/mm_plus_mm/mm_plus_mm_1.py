
class Model(torch.nn.Module):
    def forward(self, a, b, c, d, e, f):
        result = torch.mm(torch.mm(torch.mm(torch.mm(a, b), torch.mm(c, d)), e), f)
        result = result + torch.mm(torch.mm(torch.mm(a, c), e), f)
        result = result + torch.mm(torch.mm(a, c), torch.mm(d, f))
        result = result +  torch.mm(torch.mm(a, e), torch.mm(d, f))
        result = result +  torch.mm(torch.mm(e, b), torch.mm(d, f))
        result = result +  torch.mm(e, torch.mm(d, f))
        result = result +  a + b + c + d
        return result
# Inputs to the model
a = torch.randn(3, 3)
b = torch.randn(3, 3)
c = torch.randn(3, 3)
d = torch.randn(3, 3)
e = torch.randn(3, 3)
f = torch.randn(3, 3)
