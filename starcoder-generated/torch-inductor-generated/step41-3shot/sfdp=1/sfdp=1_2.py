
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat_mul = torch.nn.Linear(64, 64)
 
    def forward(self, x1, x2):
        v1 = self.mat_mul(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        v3 = v2.div(5.0)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.1)
        return v5.matmul(x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(3, 64, 48)
