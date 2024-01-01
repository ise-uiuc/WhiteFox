
class Model(torch.nn.Module):
    def forward(self, x):
        v0 = torch.mm(x, x)
        v1 = torch.mm(x, x)
        return v0 + v1
# Inputs to the model
x = torch.randn(3, 3)
