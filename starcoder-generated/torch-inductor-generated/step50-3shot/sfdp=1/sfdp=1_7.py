
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dot = torch.nn.Linear(100, 100)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x1, x2):
        v1 = self.dot(x1)
        v2 = torch.matmul(x2, v1.transpose(-2, -1))
        v3 = v2 / 1000
        v4 = torch.nn.functional.softmax(v3, -1)
        v5 = self.dropout(v4)
        v6 = torch.matmul(v5, x2)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 100)
x2 = torch.randn(128, 100, 100)
