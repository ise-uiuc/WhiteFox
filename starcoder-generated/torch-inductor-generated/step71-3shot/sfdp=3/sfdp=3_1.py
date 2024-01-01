
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(64, 64, bias=False)
        self.value = torch.nn.Linear(64, 64, bias=False)
        self.query = torch.nn.Linear(64, 64, bias=False)
        self.scale_factor = 0.67125304
        self.dropout_p = 0.6087961
    
    def forward(self, x3, x4, x5, x6, x7):
        v3 = self.key(x3)
        v4 = self.value(x4)
        v5 = self.query(x5)
        v6 = v5.bmm(v3.transpose(-2, -1))
        v7 = v6 * self.scale_factor
        v8 = torch.nn.functional.softmax(v7, dim=-1)
        v9 = v8.mul(1.0 - self.dropout_p)
        v10 = v9.bmm(v4)
        o = v10
        return o

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(128, 3, 64)
x4 = torch.randn(128, 64, 64)
x5 = torch.randn(128, 3, 64)
x6 = torch.randn(128, 64, 3)
x7 = torch.randn(3, 64)
o = m(x3, x4, x5, x6, x7)
