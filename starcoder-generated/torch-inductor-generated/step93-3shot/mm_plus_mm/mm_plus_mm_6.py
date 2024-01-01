
class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.mm(x, x)
        return torch.mm(y, x)
# Inputs to the model
x = torch.randn(2, 2)
