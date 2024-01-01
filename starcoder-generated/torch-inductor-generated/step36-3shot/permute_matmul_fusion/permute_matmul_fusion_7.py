
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(0, 2, 1)
        t2 = x2.permute(0, 2, 1)
        y1 = t2.permute(0, 2, 1)
        t3 = torch.bmm(t1, t2)
        z1 = torch.bmm(t3, y1)
        z1 = torch.matmul(y1, torch.tensor([[-1, 0, 1]]))
        z2 = torch.bmm(t3, y1)
        z2 = torch.matmul(y1, torch.tensor([[1, -2, -1]]))
        z3 = y1.detach().add(z1, alpha=2)
        return (z1, z2, z3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
