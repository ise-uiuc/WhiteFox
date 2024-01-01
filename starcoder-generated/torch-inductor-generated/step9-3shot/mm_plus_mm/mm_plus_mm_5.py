
class Model(torch.nn.Module):
    def forward(self, x, y, w, z):
        v0 = torch.mm(x, y)
        v1 = torch.mm(w, z)
        v2 = v0 - v1
        v3 = torch.tanh(v2 + v2)
        v4 = torch.mm(v3, v3)
        return v4
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
w = torch.randn(2, 2)
z = torch.randn(2, 2)
