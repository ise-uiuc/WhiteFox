
class Model(torch.nn.Module):
    def forward(self, x):
        torch.mm(x, x)
        torch.mm(x, x)
        f = lambda x: x.mm(x)
        f(x)
        return x + x
# Inputs to the model
x = torch.randn(3, 3)
