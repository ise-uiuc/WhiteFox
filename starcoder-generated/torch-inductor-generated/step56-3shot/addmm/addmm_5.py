
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, x3, x4):
        v1 = F.relu(torch.mm(x1, x2) + inp + x1)
        return F.relu(v1)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3)
inp = torch.tensor([0.25], requires_grad=True)
x3 = torch.randn(1, 3, requires_grad=True)
x4 = torch.randn(3, requires_grad=True)
