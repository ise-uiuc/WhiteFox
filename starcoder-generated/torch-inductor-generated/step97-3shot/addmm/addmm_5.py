
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x1)
        v2 = torch.mm(v1, x2)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
