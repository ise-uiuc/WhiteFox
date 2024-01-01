
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(11, 15)
        self.key = torch.nn.Linear(13, 15)
        self.value = torch.nn.Linear(13, 17)
 
    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        v3 = v1.matmul(v2.transpose(-2, -1))
        v4 = 1. / 13.
        v5 = v3 * v4
        v6 = v5.softmax(dim=-1)
        v7 = v2.matmul(v6)
        return v7

# Input for the model
x1 = torch.randn(2, 11)
x2 = torch.randn(3, 13)
