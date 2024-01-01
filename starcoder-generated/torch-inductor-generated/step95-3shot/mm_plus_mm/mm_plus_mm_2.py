
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = input1 + input2
        t2 = input3 + input4
        return t1 + t2
# Inputs to the model
input1 = torch.randn(32,64,3,24,24)
input2 = torch.randn(32,64,3,24,24)
input3 = torch.randn(32,64,3,24,24)
input4 = torch.randn(32,64,3,24,24)
