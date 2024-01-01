
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z = torch.cat([x, y], dim=-1)
        z = z.view([-1] + list(z.shape[z.dim() - 2:]))
        return z.relu()
# Inputs to the model
x = torch.randn(2, 3)
y = torch.randn(2, 4)
z = torch.randn(2, 2, 3)
a = torch.randn(3, 2)
