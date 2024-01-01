
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, t1, t2):
        t1 = t1 + x1
        return (t1 + t2) * 2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
t1 = torch.randn(3, 3, requires_grad=True)
t2 = torch.randn(3, 3, requires_grad=True)
