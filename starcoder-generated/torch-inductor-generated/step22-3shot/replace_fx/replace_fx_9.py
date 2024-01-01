
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        x2 = torch.rand(5, 7, 8, device=x1.device)
        x3 = torch.rand(5, 7, 8, device=x1.device)
        x4 = x2 + x3
        x5 = torch.rand_like(x4, device=x1.device)
        x6 = x4 + x5
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 3)
