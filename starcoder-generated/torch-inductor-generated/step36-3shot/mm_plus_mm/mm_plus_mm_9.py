
class Model(torch.nn.Module):
    def forward(self, x):
        res = 0.
        x = x * tensor(7)
        v1 = torch.mm(x, x)
        return v1.flatten()
# Inputs to the model
x = torch.randn(1, 1)
