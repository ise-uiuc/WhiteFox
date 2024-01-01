
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        f1 = torch.ones((1, 4, 4))
        f2 = torch.ones((1, 4, 4))
        v1 = torch.matmul(x1, x2)
        v2 = torch.matmul(x2, x1)
        return [v1, v2, v1, f1, v1, v1, f2]
# Inputs to the model
x1 = torch.ones(1, 4, 4)
x2 = torch.ones(1, 4, 4)
