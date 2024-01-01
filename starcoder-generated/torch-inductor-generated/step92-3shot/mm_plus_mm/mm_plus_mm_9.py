
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, t1)
        t3 = torch.mm(input3, t1)
        return t2 + t3
# Inputs to the model
input1 = torch.randn(32, 32)
input2 = torch.randn(32, 32)
input3 = torch.randn(32, 32)
