
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.nn.functional.batch_norm(x1, train=True)
        t2 = t1 + x2
        t3 = t2 + x2
        x = x + t3
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = 1
