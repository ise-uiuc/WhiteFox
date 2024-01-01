
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat(
            [x, torch.rand_like(x).detach().requires_grad_(True)])
        x = x ** 2
        x = x.relu()
        return x
# Inputs to the model
x = torch.randn(3, requires_grad=True)
