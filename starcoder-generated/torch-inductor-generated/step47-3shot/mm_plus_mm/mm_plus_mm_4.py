
class Model(torch.nn.Module):
    def __init__(self, input1, input2, input3):
        super().__init__()
        self.weight = torch.randn(input1, input2)
        self.bias = torch.randn(input2)
        self.dummy1 = torch.randn(2, 2)
        self.dummy2 = torch.randn(3, 3)
    def forward(self, input4, input5, input6):
        tmp1 = torch.mm(self.weight, input4) + self.bias
        tmp2 = torch.mm(self.dummy1, self.dummy2)
        return torch.mm(tmp1, input5) + torch.mm(tmp2, input6)
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
input5 = torch.randn(5, 5)
input6 = torch.randn(5, 5)
