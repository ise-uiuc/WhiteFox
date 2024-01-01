
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = t1 + t1
        t3 = torch.mm(input, input)
        t4 = t2 + t3
        t5 = t4 + t3
        t6 = torch.mm(input, input)
        t7 = t5 + t6
        return t7
# Inputs to the model
input = torch.randn(16, 16)
