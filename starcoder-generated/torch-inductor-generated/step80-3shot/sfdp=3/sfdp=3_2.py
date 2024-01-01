
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.functional.linear
        self.dot = torch.nn.functional.linear
 
    def forward(self, x1, x2):
        v1 = self.matmul(x1, x2)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v1.softmax()
        v5 = torch.nn.functional.dropout(v4, p=0.1)
        v6 = self.dot(v5, x2)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
