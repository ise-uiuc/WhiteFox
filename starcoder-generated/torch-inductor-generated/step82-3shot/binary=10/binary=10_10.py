
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2
 
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = v2 + x1
        return v3
 
input_tensor1 = torch.randn(4)
input_tensor2 = torch.randn(4)
m1 = Model1()
m2 = Model2()
__output_m1__ = m1(input_tensor1, input_tensor2)
__output_m2__ = m2(input_tensor1)

