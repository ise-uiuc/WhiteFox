
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t2 = torch.mm(input1, input4)
        t3 = torch.mm(input3, input2)
        t4 = torch.mm(input1, input3)
        t6 = t2 + t3
        t7 = t4 + t2
        t8 = t6 * t7
        return t8
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
