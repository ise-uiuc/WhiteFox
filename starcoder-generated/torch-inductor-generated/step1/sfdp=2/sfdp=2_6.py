
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(8, 8, bias=True)
        self.query = torch.nn.Linear(8, 8, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x, dropout_p):
        v1 = torch.matmul(self.query(x), self.key(x).transpose(-2, -1))
        v2 = v1 / x.shape[-1]
        v3 = self.softmax(v2)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, x)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.rand(1, 8, 20)
dropout_p = torch.rand((x.shape[0], x.shape[2]))
