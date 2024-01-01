
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(10, 10)
        self.m2 = torch.nn.Linear(10, 10)
        
    def forward(self, inp):
        v0 = self.m1(inp)
        v1 = torch.matmul(v0, v0.transpose(-2, -1))
        v2 = v1.div(2)
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, 0.5)
        v5 = torch.matmul(v4, v0)
        v6 = self.m2(v5)
        return v6
        
# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 10)
