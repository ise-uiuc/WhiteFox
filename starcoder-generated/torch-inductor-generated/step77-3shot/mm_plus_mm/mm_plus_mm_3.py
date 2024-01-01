
class Model(torch.nn.Module):
    def forward(self, x, y):
        z = torch.mm(x, y) + torch.mm(y, x)
        return z
# Inputs to the model
x = torch.randn(5, 5)
y = torch.randn(5, 5)
