
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Embedding(10, 50)
        self.key = torch.nn.Embedding(10, 50)
        self.value = torch.nn.Embedding(10, 50)
 
    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.key(x1)
        v3 = self.value(x1)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4 / 0.5
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=0.3)
        v8 = torch.matmul(v7, v3)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randint(10, (1, 5,))

