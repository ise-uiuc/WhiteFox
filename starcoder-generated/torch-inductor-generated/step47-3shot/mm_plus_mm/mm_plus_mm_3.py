
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input2, input2)
        t3 = torch.mm(input3, input3)
        t4 = torch.mm(input4, input4)
        t5 = t3 - t4
        t6 = torch.mm(input4, input4)
        return t1 + t2 - t5 - t6
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
