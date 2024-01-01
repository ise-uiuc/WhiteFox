
class Model(torch.nn.Module):
    def __init__(self, N, D):
        super().__init__()
        self.N = N
        self.d1 = torch.nn.Linear(D, D)
        self.d2 = torch.nn.Linear(D, D)
        self.d3 = torch.nn.Linear(D, D)
 
    def forward(self, x1, x2, x3):
        v1 = self.d1(x1)
        v2 = torch.matmul(x2, v1.transpose(-2, -1))
        v3 = v2 * self.N ** -0.5
        v4 = nn.functional.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, x3)
        v6 = self.d2(v5)
        v7 = self.d3(v6)
        return v7

# Initializing the model
N = 2
D = 512
m = Model(N, D)

# Inputs to the model
x1 = torch.randn(32, 512)
x2 = torch.randn(N, D, 32)
x3 = torch.randn(N, D, 32)
