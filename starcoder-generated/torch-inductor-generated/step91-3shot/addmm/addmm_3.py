
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = t1 + input1
        return t2
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
