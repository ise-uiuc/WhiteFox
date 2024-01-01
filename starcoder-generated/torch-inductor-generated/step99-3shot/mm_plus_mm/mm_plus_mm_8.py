
class Model(torch.nn.Module):
    def forward(self, x, y):
        o1 = torch.mm(x, y)
        o2 = torch.mm(y, o1)
        o3 = torch.mm(o2, o1)
        o4 = torch.mm(o3, o1)
        o5 = torch.mm(x, o4)
        return o5
# Inputs to the model
x = torch.randn(4, 4)
y = torch.randn(4, 4)
