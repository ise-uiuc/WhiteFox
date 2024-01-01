
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x3)
        v2 = torch.mm(x2, x4)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.FloatTensor(5, 3, 7)
x2 = torch.FloatTensor(5, 7, 6)
x3 = torch.FloatTensor(5, 3, 7)
x4 = torch.FloatTensor(5, 7, 6)
