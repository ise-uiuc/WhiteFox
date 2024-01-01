
class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear_1 = torch.nn.Linear(self.input_dim, self.input_dim)
 
    def forward(self, x1, x2):
        v1 = self.linear_1(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        v3 = v2.div(1 / math.sqrt(64))
        v4 = torch.nn.functional.softmax(v3, dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.2)
        v6 = torch.matmul(v5, x2)
        return v6

# Initializing the model
m = Model(64)

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 100, 64)
