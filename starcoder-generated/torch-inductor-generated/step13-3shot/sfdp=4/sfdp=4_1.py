
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(32, 32)
        self.linear1 = torch.nn.Linear(32, 32)

    def forward(self, x1):
        mat0 = self.linear0(x1)
        mat1 = self.linear1(x1)
        m0 = mat0.matmul(mat1.transpose(-2, -1)) / math.sqrt(mat0.size(-1))
        m1 = m0 + (torch.ones(32, 32) * 100000)
        w12 = torch.softmax(m1, dim=-1)
        o12 = w12.matmul(mat0)
        return o12

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 32)
