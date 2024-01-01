
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        v3 = v2.div(1e-5)
        v4 = nn.functional.softmax(v3, dim=-1) + nn.functional.dropout(v4, p=0.3)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 4, 100)
x2 = torch.randn(5, 7, 100)
x3 = torch.randn(5, 7, 100)
