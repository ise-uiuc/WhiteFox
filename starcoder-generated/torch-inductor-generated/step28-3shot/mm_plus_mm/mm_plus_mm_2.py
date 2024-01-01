
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input4, input2)
        t3 = torch.mm(input3, input2)
        t4 = torch.mm(input2, input4)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
