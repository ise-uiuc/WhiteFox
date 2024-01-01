
class Model(nn.Module):
    def forward(self, w, x, y):
        z = torch.mm(w, x) + torch.mm(x, w)
        a = torch.mm(y, z) + torch.mm(z, y)
        return torch.tanh(a)
# Inputs to the model
w = torch.randn(7, 7)
x = torch.randn(7, 7)
y = torch.randn(7, 7)
