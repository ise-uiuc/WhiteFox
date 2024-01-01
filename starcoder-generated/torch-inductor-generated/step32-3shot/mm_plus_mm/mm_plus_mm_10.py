
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = torch.mm(x, x)
        v3 = v1 + v2
        v4 = v1 - v2
        return v3 + v4
# Inputs to the model
x = torch.randn(5, 5)
