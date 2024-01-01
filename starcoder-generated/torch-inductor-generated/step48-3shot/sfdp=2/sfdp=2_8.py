
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.nn.Embedding()
        self.linear_in = torch.nn.Linear()
        self.ln1 = torch.nn.LayerNorm()
        self.ln2 = torch.nn.LayerNorm()
        self.linear_out = torch.nn.Linear()
        self.dropout = torch.nn.Dropout()
 
    def forward(self, x1, x2):
        v1 = self.embedding()
        v2 = self.linear_in(x1)
        v3 = self.ln1(v2)
        v4 = self.ln2(v2)
        v5 = torch.matmul(v3, v4.transpose(-2, -1))
        v6 = v5.div(0.06)
        v7 = v6.softmax(dim=-1)
        v8 = F.dropout(v7, 0.15, True)
        v9 = torch.matmul(v8, x2)
        v10 = v9.norm()
        return x1, x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(5, 4, 64, 64)
