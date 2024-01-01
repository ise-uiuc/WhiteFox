
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input1)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
