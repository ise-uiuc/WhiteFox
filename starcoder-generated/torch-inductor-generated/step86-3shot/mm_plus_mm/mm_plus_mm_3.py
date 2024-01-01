
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input1)
        t3 = torch.mm(input2, input2)
        t4 = torch.mm(input1, input2)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
