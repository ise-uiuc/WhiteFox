
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input4)
        t = torch.mm(input4, input2)
        t = t + t1
        t = torch.mm(input3, input2)
        t = t + t
        return t
# Inputs to the model
input1 = torch.randn(5, 8)
input2 = torch.randn(8, 16)
input3 = torch.randn(16, 5)
input4 = torch.randn(5, 16)
