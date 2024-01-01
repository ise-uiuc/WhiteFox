
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x.view(x.shape[0], 10, -1)
        w = z.unsqueeze(-1)
        u = z.permute(1, 2, 0)
        v = u.view(x.shape[0], -1)
        return v.relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
