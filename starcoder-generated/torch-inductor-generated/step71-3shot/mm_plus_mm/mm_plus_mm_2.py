
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        s = x1 + x2
        t = torch.mm(s, s)
        return torch.mm(t, x1) + torch.mm(t, x2)
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
