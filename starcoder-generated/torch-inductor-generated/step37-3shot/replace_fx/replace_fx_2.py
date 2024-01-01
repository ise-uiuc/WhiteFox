
class Model(torch.nn.Module):
    def forward(self, x):
        p = torch.rand((1), 5).expand(100)
        return torch.add(x, p)
# Inputs to the model
x = torch.randn(1, 5)
