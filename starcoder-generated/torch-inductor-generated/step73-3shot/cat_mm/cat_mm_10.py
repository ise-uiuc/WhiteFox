
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = torch.einsum('ai,bi->ab', (x1, x2))
        b = torch.einsum('ai,bi->ab', (x1, x2))
        c = torch.einsum('ai,bi->ab', (x1, x2))
        d = torch.einsum('ai,bi->ab', (x1, x2))
        e = torch.einsum('ai,bi->ab', (x1, x2))
        return torch.cat([a, b, c, d, e], 1)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
