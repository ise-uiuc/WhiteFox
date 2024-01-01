
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        v1 = torch.add(torch.mm(x, t2), p)
        return v1
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
t2 = torch.randn(3, 3)
p = torch.randn(3, 3, requires_grad=True)
