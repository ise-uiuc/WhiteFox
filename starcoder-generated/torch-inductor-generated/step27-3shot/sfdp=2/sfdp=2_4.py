
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_linear = torch.nn.Linear(3, 8)
        self.key_linear = torch.nn.Linear(3, 8)
        self.value_linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        q1 = self.query_linear(x1)
        k1 = self.key_linear(x1)
        v1 = self.value_linear(x1)
        q2 = self.query_linear(x2)
        k2 = self.key_linear(x2)
        v2 = self.value_linear(x2)
        q3 = torch.matmul(q1, k2.transpose(-2, -1))
        k3 = torch.matmul(q2, k1.transpose(-2, -1))
        v3 = torch.matmul(q2, v1.transpose(-2, -1))
        v4 = torch.matmul(q1, v2.transpose(-2, -1))
        q4 = q3 + k3
        k4 = q4 + k3
        v5 = v3 + v4
        v6 = torch.matmul(q2, v5.transpose(-2, -1))
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
x2 = torch.randn(1, 3, 4, 4)
