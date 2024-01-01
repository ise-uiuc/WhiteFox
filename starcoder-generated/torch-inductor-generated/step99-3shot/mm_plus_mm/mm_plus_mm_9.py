
class Model(torch.nn.Module):
    def forward(self, x, y):
        t0 = torch.mm(x, y)
        t1 = torch.mm(y, y)
        return t0 + t1
# Inputs to the model
x = torch.randn(2, 3)
y = torch.randn(3, 3)
