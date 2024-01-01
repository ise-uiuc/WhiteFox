
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.mm(torch.mm(x, x), x)
# Inputs to the model
x = torch.randn(8, 8)
