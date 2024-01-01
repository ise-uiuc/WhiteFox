
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 16)
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = self.fc(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3 / 2.0
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 8)
x2 = torch.randn(16, 8)
