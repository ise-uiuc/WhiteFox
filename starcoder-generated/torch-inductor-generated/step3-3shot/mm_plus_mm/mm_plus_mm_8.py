
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1.transpose(1, 0), input2)
        t2 = torch.mm(input3, input4)
        output = t1 + t2
        return output
# Inputs to the model
input1 = torch.randn(28, 28)
input2 = torch.randn(28, 28)
input3 = torch.randn(28, 28)
input4 = torch.randn(28, 28)
