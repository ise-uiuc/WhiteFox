
class Model(nn.Module):
    def forward(self, x, y, z):
        t1 = torch.mm(x, y)
        t2 = torch.mm(t1, z)
        t3 = torch.tanh(t2)
        t4 = torch.mm(y, z)
        t5 = torch.tanh(t4)
        return t3 + t5
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
