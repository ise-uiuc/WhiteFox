
class Model(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
    def forward(self, input, hidden):
        t1 = torch.nn.ReLU()(hidden)
        t2 = torch.mm(input, t1)
        t3 = t1 + t2
        t4 = torch.mm(input, t2)
        t5 = t3 + t4
        return t5, t4
# Inputs to the model
input1 = torch.randn(100, 100)
input2 = torch.randn(100)
