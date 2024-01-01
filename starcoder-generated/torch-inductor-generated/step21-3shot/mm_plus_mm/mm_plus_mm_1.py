
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input3, input4)
        t2 = torch.mm(input1, input4)
        t2 = torch.mm(input1, input2)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(16, 8)
input3 = torch.randn(2, 8)
input4 = torch.randn(8, 2)
