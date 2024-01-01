
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.add(x2, inp)
        x1.retain_grad()
        return torch.mm(x1, v1) # This line should change to the two lines below if x1 is the desired tensor
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
