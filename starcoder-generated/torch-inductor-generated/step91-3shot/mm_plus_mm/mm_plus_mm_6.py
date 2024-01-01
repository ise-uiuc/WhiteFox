
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        h1 = torch.mm(x1, x2)
        return h1 + torch.mm(x2.t(), x1.t())
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
