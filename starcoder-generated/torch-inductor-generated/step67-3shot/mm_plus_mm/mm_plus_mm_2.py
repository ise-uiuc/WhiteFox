
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2):
        input1 = torch.mm(x1, x2)
        input2 = torch.mm(x1, x2)
        return input1 + input2
# Inputs to the model
x1 = torch.randn(100, 100)
x2 = torch.randn(100, 100)
