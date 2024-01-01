
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1613765264515159)
        self.linear1 = torch.nn.Linear(3, 32)
        self.linear2 = torch.nn.Linear(32, 10)
        
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = self.dropout(v2)
        v4 = self.linear1(x2)
        v5 = v4.transpose(-2, -1)
        v6 = torch.matmul(v3, v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 3)
x2 = torch.randn(10, 3)
