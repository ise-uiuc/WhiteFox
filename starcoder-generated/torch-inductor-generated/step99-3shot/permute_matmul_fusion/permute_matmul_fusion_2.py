
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        res = []
        for _ in range(3):
            x2 = x2.permute(0, 2, 1)
            res.append(torch.matmul(x1, x2))
        return res
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
