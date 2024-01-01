
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(t2, input3)
        t4 = torch.mm(t3, input2)
        t5 = input3 + t4
        t6 = torch.mm(t4, input1)
        out = input2 + t5 + t6
        return out
    
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
