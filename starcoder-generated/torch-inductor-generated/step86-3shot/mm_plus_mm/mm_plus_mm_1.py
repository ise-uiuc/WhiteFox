
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input2)
        return torch.mm(t1, t2)
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
