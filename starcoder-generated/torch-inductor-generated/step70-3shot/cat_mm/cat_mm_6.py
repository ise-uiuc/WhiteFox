
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy input tensor is created here
        torch.randn(10, 143, 600, 800)
    def forward(self, x1, x2):
        # Dummy input tensor is used here
        v1 = torch.randn(4, 32)
        v2 = torch.randn(2, 16)
        v3 = torch.randn(10, 143, 600, 800)
        v4 = torch.cat([v1, v1, v1, v1], 1)
        v5 = torch.cat([v2, v2], 1)
        return torch.cat([v3, v3, v3, v3, v4, v4, v4, v4, v4, v5, v5, v5, v5, v5], 1)
# Inputs to the model
x1 = torch.randn(8, 16)
x2 = torch.randn(16, 32)
