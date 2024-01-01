
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(1, 1)
        self.w2 = torch.randn(1, 1)
        self.w3 = torch.randn(1, 1)
def forward(self):
        t1 = torch.mm(self.w1, self.w2)
        t2 = torch.mm(self.w1, self.w3)
        t3 = torch.mm(self.w2, self.w3)
        return t1 + t2 + t3 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(4, 4)
        self.w2 = torch.randn(4, 4)
        self.w3 = torch.randn(4, 4)
        self.w4 = torch.randn(4, 4)
        self.w5 = torch.randn(4, 4)
        self.w6 = torch.randn(4, 4)
def forward(self):
        t1 = torch.mm(self.w1, self.w2)
        t2 = torch.mm(self.w1, self.w2)
        t3 = torch.mm(self.w1, self.w3)
        t4 = torch.mm(self.w4, self.w5)
        t5 = torch.mm(self.w3, self.w4)
        t6 = torch.mm(self.w4, self.w6)
        t7 = torch.mm(self.w5, self.w6)
        return t1 + t2 + t3 + t4 + t5 + t6 + t7
# Inputs to the model
