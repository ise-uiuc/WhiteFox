
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Matmul(2, 3)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.matmul2 = torch.nn.Matmul(3, 4)
 
    def forward(self, x1, x2):
        v1 = self.matmul1(x1, x2)
        v2 = v1 * 0.5
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        v5 = self.matmul2(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 4)
