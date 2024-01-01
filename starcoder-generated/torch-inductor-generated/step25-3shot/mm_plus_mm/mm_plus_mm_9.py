
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(2, 2)
    def forward(self, input1, input2, input3, input4):
        v1 = torch.mm(input1, self.w1)
        v2 = torch.mm(input2, self.w1)
        v3 = torch.mm(input3, self.w1)
        v4 = torch.mm(input4, self.w1)
        t1 = torch.mm(v2, v1)
        t2 = torch.mm(v1, v3)
        t3 = torch.mm(v4, v3)
        return t1 + t2 + t3
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
