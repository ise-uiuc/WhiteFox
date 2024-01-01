
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = torch.mm(x, x)
        return v1 + v2
# Inputs to the model
x = torch.randn(5, 5)
