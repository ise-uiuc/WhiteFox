
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(512, 512)
        self.matmul2 = torch.nn.Linear(512, 512)
 
    def forward(self, x1):
        v1 = torch.matmul(x1, self.matmul1(x1))
        v2 = v1/100
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        v5 = torch.matmul(v4, self.matmul2(v4))
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(512, 512)
x2 = torch.randn(512, 512)
