
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = torch.mm(x, x)
        t = v1 + v2
        v3 = torch.mm(x, x)
        return t * v3
# Inputs to the model
x = torch.randn(100, 100)
