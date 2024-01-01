
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 7)
        self.linear2 =  torch.nn.Linear(4, 4)
 
    def forward(self, x1, x2):
        w1 = self.linear1(x1)
        w2 = self.linear2(x2)
        v1 = torch.matmul(w1, w2.transpose(-2, -1))
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 4, 64, 64)
