

t2 = torch.mul(in1, in2)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input3, input4):
        return torch.cat([t2, torch.mm(input3, input4)],1)
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(1, 2)
input3 = torch.randn(2, 1)
input4 = torch.randn(1, 2)
