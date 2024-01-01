
class Model(torch.nn.Module):
    def __init__(self, arg1, arg2):
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg1_1 = arg1.conv
        self.arg2_1 = arg2.conv
 
    def forward(self, x, inp1=None, inp2=None):
        v1 = self.arg1_1(x)
        v2 = self.arg2_1(x)
        v3 = v1 + v2
        if (inp1 is None) or (inp2 is None):
            v6 = v3 + inp1 + inp2
            return v6
        else:
            v4 = self.arg1(x, inp1, inp2)
            v5 = self.arg2(x, inp1, inp2)
            v6 = v4 + v5 + inp1 + inp2
            return v6

# Initializing the model
arg1_1 = Model(m, m)
arg2_1 = arg1_1
arg1 = arg1_1
arg2 = arg2_1
m = Model(arg1, arg2)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
