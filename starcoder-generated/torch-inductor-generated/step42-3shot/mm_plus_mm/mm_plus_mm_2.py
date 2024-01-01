
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        res1 = t1 + t2
        return res1.mm(t2)
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
