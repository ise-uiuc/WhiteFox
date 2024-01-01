
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(16, 16, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.dropout(v1)
        v3 = self.linear2(v2)
        v4 = torch.matmul(v3, x2.transpose(-2, -1))
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
x2 = torch.randn(1, 64, 40)
