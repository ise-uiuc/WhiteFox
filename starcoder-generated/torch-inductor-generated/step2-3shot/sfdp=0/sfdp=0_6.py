
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(96, 96, bias=False)
        self.key = torch.nn.Linear(96, 32, bias=False)
        self.value = torch.nn.Linear(96, 32, bias=False)

    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.key(x1)
        v3 = v2.transpose(-2, -1)
        v4 = torch.matmul(v1, v3)
        v5 = 1.0 / math.sqrt(v4.shape[-1])
        v6 = torch.softmax(v4 * v5, -1)
        v7 = self.value(x1)
        v8 = torch.matmul(v7, v6.transpose(-2, -1))
        return v8

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(2, 5, 96)
