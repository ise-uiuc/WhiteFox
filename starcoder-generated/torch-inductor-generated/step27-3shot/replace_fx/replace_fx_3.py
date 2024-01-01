
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.rand_like(x)
        a2 = torch.rand_like(x)
        a3 = 0.3752 * a1 - 0.2977 * a2
        return 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
