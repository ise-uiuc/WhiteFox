
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.mul(x1.softmax(dim = 1), torch.mul(x1.softmax(dim = 1).softmax(dim = 1), torch.mul(x1.softmax(dim = 1).softmax(dim = 1).softmax(dim = 1), x1.softmax(dim = 1).softmax(dim = 1).softmax(dim = 1).softmax(dim=1))))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
