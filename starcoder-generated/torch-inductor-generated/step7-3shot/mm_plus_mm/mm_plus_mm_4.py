
class Model(torch.nn.Module):
    def forward(self, x):
        i1 = torch.mm(x,x)
        i2 = torch.mm(x,x)
        i3 = torch.mm(x,x)
        i4 = i1 + i2 + i3 + i2
        return i4
# Inputs to the model
x = torch.randn(10, 5)
