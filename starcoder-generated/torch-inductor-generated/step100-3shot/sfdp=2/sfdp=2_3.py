
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(64, 32)
        self.key = torch.nn.Linear(64, 32)
        self.scale_factor = torch.sqrt(torch.FloatTensor([64.0]))
        self.dropout = torch.nn.Dropout(0.33)
 
    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.dropout(self.key(x1))
        v3 = torch.matmul(v1, v2.transpose(-1, -2))
        v4 = v3.div(self.scale_factor)
        v5 = F.softmax(v4, dim=-1)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, x1)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
