
class Model(torch.nn.Module):
    def forward(self, x, v1):
        y = torch.mm(v1,x) + torch.mm(x,v1)
        return y
# Inputs to the model
x = torch.randn(4, 4)
v1 = torch.randn(4, 4)
