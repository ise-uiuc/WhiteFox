
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(192, 128)
        self.key = torch.nn.Linear(192, 256)
        self.value = torch.nn.Linear (384, 256)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x1, x2, x3):
        q = self.query(x1) # Query
        k = self.key(x2) # Key
        v = self.value(x3) # Value
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(256)
        att = self.dropout(torch.softmax(att, dim=-1))
        return torch.matmul(att, v)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 192)
x2 = torch.randn(4, 384)
x3 = torch.randn(100, 256)
dropout_p = 0.1
