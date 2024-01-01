
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([torch.mm(x1[i], x2[i]) for i in range(10)], 1)
# Inputs to the model
x1 = [torch.randn(10, 512) for _ in range(5)]
x2 = [torch.randn(512, 4) for _ in range(5)]
