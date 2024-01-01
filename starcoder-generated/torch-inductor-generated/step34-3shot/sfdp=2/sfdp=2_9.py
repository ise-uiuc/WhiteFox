
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 24)
        self.fc2 = torch.nn.Linear(24,8)
 
    def forward(self, x1, x2):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = torch.matmul(x2, v2.transpose(1,0))
        v4 = v3.softmax(dim=-1)
        v5 = v4.dropout(p=0)
        return v5.matmul(v1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10, 1)
x2 = torch.randn(20, 10, 1)
