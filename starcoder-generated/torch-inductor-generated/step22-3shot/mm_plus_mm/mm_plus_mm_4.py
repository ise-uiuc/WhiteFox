
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.randn(5, 5)
        self.t2 = torch.randn(5, 5)
    def forward(self, input1, input2):
        t1 = input1.mm(input2.mm(self.t1))
        mmm = t1.mm(self.t1)
        mm = input2.mm(input1.mm(self.t1.mm(self.t2)))
        return t1 + mmm - mm
# Inputs to the model
input1 = torch.randn(8, 8)
self.t1 = torch.randn(8, 8)
self.t2 = torch.randn(8, 8)
