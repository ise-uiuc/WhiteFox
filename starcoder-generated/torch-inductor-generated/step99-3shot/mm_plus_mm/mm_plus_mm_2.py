
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        out=torch.mm(x1, x2)
        out=torch.mm(x2, x1)
        out=torch.mm(x1, x2)
        return out
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
