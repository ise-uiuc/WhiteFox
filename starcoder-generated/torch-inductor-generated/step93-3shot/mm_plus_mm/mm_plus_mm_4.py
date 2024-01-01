
class Model(torch.nn.Module):
    def forward(self, x):
        x = torch.mm(x, x)
        x = torch.mm(x, x)
        return x
# Inputs to the model
x = torch.randn(6, 6)
