
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        t1 = torch.mm(x, z)
        t2 = torch.mm(y, z)
        k = t2.unsqueeze(0)
        l = t1.unsqueeze(0)
        m = k * l
        return m
# Inputs to the model
x = torch.randn(3, 32)
y = torch.randn(3, 32)
z = torch.randn(32, 32)
