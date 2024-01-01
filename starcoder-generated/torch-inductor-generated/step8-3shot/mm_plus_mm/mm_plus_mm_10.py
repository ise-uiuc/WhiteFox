
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input4)
        t3 = torch.mm(input3, input2)
        t4 = torch.mm(input3, input4)
        t5 = t1 + t2
        t6 = t3 + t4
        return t5 * t6
# Inputs to the model
input1 = torch.randn(10, 3)
input2 = torch.randn(3, 10)
input3 = torch.randn(10, 7)
input4 = torch.randn(7, 10)
