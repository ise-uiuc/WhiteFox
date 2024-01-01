
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(4, 4, 64, 64))
        inq = torch.randn(4, 4, 64, 64)
        torch.nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))
        self.padding = torch.nn.ConstantPad2d(1, -1e9)
        self.value = torch.nn.Parameter(torch.empty(64, 32, 64, 64))
        inv = torch.randn(64, 32, 64, 64)
        torch.nn.init.kaiming_normal_(self.value, a=math.sqrt(5))

    def forward(self, x1):
        q = torch.matmul(self.query, x1.transpose(-2, -1))
        s1 = q.div(32)
        w1 = torch.nn.functional.softmax(s1, dim=-1)
        d1 = torch.nn.functional.dropout(w1, 0.2)
        v = torch.matmul(d1, self.value.transpose(-2, -1))
        return v 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
