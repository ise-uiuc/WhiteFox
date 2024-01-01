
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.mm(x, x)
# Inputs to the model
x = torch.randn(150, 75)
