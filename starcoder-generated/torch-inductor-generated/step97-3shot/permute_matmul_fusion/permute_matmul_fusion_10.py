
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(0, 2, 1)
        t2 = x2.permute(-1, 1)
        return torch.bmm(t1, t2)
# Inputs to the model
x1 = torch.randn(2, 6, 4)
x2 = torch.randn(1, 4, 5)
