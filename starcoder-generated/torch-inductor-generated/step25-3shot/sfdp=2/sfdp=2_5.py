
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(12, 33)
        self.matmul2 = torch.nn.Linear(33, 44)
 
    def forward(self, x1, x2):
        v1 = self.matmul1(x1)
        v2 = self.matmul2(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.div(10)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.2)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
x2 = torch.randn(1, 8, 2, 2)
