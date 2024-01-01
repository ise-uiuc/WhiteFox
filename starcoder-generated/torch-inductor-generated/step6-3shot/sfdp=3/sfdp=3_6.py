
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, v1, v2):
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.mul(0.5)
        v5 = self.softmax(v4)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 100, 10)
v2 = torch.randn(1, 100, 100)
