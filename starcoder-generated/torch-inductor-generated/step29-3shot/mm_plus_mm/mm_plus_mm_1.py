
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input3, input1)
        t2 = torch.mm(input1, input1)
        t3 = torch.mm(input2, input3)
        t4 = t1 + t2
        return t4 - t3
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
