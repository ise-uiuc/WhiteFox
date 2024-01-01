
class Model(torch.nn.Module):
    def forward(self, x, y, z, w):
        T1 = torch.mm(x, y)
        T2 = torch.mm(w, x)
        return T1 + T2
# Inputs to the model start
x = torch.randn(10, 11)
y = torch.randn(11, 12)
z = torch.randn(11, 12)
w = torch.randn(10, 12)
