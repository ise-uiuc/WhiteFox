
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1.transpose(1, 0), x3)
        v2 = torch.mm(x1, x3.transpose(1, 0))
        return v1 + v2
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
x3 = torch.randn(3, 3)
