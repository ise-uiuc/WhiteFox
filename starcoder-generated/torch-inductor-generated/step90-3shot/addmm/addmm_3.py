
class Model(torch.nn.Module):
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = inp + v1
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
