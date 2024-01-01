
class Model(torch.nn.Module):
    def forward(self, w, x, y):
        z = torch.mm(w, x) + torch.mm(x, y)
        return z
# Inputs to the model
w = torch.randn(5, 5)
x = torch.randn(5, 5)
y = torch.randn(5, 5)
