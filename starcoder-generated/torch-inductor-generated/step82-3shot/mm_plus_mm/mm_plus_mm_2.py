
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.matmul
    def forward(self, input1, input2, input3, input4):
        t1 = self.t1.forward(input1, input4)
        t2 = self.t1.forward(input3, input2)
        t3 = self.t1.forward(input2, input3)
        return t1 + t2 + t3 
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.mm
    def forward(self, input1, input2, input3, input4):
        t1 = self.t1.forward(input1, input4)
        t2 = self.t1.forward(input3, input2)
        t3 = self.t1.forward(input2, input3)
        return t1 + t2 + t3 
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
