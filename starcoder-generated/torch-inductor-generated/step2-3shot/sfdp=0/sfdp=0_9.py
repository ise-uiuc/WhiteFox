
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(25, 10)
        self.key = torch.nn.Linear(10, 10)
        self.query = torch.nn.Linear(10, 10)
        self.value = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = self.linear(x2)
        v3 = v1 + v2 # Concatenate the inputs by the second dimension (i.e. the number of columns)
        v4 = self.key(v3)
        v5 = self.query(v3)
        v6 = self.value(v3)
        v7 = torch.matmul(v4, v5.transpose(-2, -1)) # Compute the scaled dot-product attention weights
        v8 = torch.nn.functional.softmax(v7, dim=-1)
        v9 = torch.matmul(v8, v6)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25)
x2 = torch.randn(1, 25)
