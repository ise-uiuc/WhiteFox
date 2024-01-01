
class Model(nn.Module):
    def __init__(self, input1):
        super(Model, self).__init__()
    def forward(self, input2, input3):
        t1 = torch.mm(input2, input3)
        t2 = torch.mm(input1, input3)
        return t1 + t2        
# Inputs to the model
input1 = torch.randn(31, 31)
input2 = torch.randn(31, 31)
input3 = torch.randn(31, 31)
