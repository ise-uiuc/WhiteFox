
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(2, 1)
    def forward(self, input1, input2):
        t1 = torch.mm(input1, self.weight.mm(input2))
        t1 = torch.mm(t1, self.weight)
        return torch.cat([t1, t1, t1], 1)
# Inputs to the model
input1 = torch.randn(2, 3)
input2 = torch.randn(3, 2)
