
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = torch.where(x1 < 0.1, x1, torch.rand_like(x1))
        x1 = x1 + a
        return x1
# Inputs to the model
x1 = torch.randn(1,2, 2)
