
class Model(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.mul(0.125)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 75, 64)
x2 = torch.randn(1, 60, 64)
