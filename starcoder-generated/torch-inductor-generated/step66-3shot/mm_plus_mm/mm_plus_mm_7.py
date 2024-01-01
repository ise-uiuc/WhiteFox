
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input):
        t1 = torch.mm(input, torch.ones(100, 100))
        return t1 + t1
# Inputs to the model
input = torch.randn(10000, 10000)
