
class Model(torch.nn.Module):
    def __init__(self, input1, input2, input3, input4):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(input1.size(), requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.randn(input2.size(), requires_grad=True))
        self.w3 = torch.nn.Parameter(torch.randn(input3.size(), requires_grad=True))
        self.w4 = torch.nn.Parameter(torch.randn(input4.size(), requires_grad=True))
    def forward(self):
        v1 = torch.mm(self.w1, self.w2)
        v2 = torch.mm(self.w3, self.w4)
        v3 = v1 + v2
        return v3
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
