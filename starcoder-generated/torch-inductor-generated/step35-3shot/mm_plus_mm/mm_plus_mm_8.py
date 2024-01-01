
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x2, x1)
        return (torch.mm(t1, t2))
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
