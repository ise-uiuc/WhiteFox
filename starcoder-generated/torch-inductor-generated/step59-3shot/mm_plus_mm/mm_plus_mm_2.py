
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input4)
        t3 = torch.mm(input5, input6)
        t4 = t1 + t2
        return t3 + t4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_module = CustomModule()
    
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, input6)
        t4 = t1 + t2
        return t3 + t4
# Inputs to the model
input1 = torch.randn(20, 20)
input2 = torch.randn(20, 20)
input3 = torch.randn(20, 20)
input4 = torch.randn(20, 20)
input5 = torch.randn(20, 20)
input6 = torch.randn(20, 20)
