
class myModel(nn.Module):
    def __init__(self, input):
        super(myModel, self).__init__()
        self.input = input
    def forward(self, x1, x2):
        v1 = torch.mm(x1, self.input)
        v2 = torch.mm(x2, self.input)
        return v1 * v2

input = torch.randn(7, 9)
m = myModel(input)
# Inputs to the model
x1 = torch.randn(9, 9)
x2 = torch.randn(9, 9)
