
class Model(torch.nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.m = 10
        self.l = [self.m, 2, 10] 
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(self.s, p = 0.5)
        x3 = torch.nn.functional.dropout(self.l, p=0.5)
        x3 += 2
        return x2
# Inputs to the model
x1 = torch.randn(3, 4)
