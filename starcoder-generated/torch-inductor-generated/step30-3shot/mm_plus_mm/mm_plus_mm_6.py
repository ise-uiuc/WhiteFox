
class Model(torch.nn.Module):
    def forward(self, t1, t2):
        a = torch.mm(t1, t2)
        b = torch.mm(t2, t1)
        return torch.mm(a, b)
# Inputs to the model
x = torch.randn(1, 1)
y = torch.randn(1, 1)
z = torch.randn(1, 1)
