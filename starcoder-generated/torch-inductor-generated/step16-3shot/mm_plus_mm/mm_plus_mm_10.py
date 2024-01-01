
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.randn(3, 3)
        self.w2 = torch.randn(3, 3)
        self.w3 = torch.randn(3, 3)
        self.w4 = torch.randn(3, 3)
    def forward(self, input):
        t1 = torch.mm(input, self.w1)
        t2 = t1*torch.mm(input, self.w2) + torch.mm(input, self.w3)
        t3 = t1*torch.mm(input, self.w2) + torch.mm(input, self.w4)
        t4 = t1*torch.mm(t2, self.w2) + torch.mm(input, self.w3)
        return t4
# Inputs to the model
input = torch.randn(10, 10)
