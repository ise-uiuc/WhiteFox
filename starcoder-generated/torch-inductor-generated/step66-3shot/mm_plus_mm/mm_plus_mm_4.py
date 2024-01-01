
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        t4 = t1 * t2
        t5 = t2 * t3
        t6 = t3 * t1
        t7 = t1 * t2 + t4 + t5 + t6
        return t7
# Inputs to the model
input = torch.randn(16, 16)
