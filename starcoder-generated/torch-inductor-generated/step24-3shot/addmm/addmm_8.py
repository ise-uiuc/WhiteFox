
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2, inp3):

        v1 = torch.mm(x1.transpose(0, 1), inp1)
        v2 = torch.mm(x1, v1) + torch.mm(x2, inp2)
        v3 = v2 + inp3
        return v3
# Inputs to the model
x1 = torch.randn(1024, 1321)
x2 = torch.randn(1024, 1)
inp1 = torch.randn(1321, 1)
inp2 = torch.randn(1, 1)
inp3 = torch.randn(1, 376)
