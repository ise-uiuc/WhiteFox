
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(30, 15)
        self.linear2 = torch.nn.Linear(15, 20)
 
    def forward(self, query, key, value, dropout_p):
        v1 = self.linear1(query)
        v2 = self.linear2(key)
        v3 = torch.matmul(v1, v2.transpose(0, 1))
        v4 = v3.div(5.0)
        v5 = v4.softmax(dim=-1)
        v6 = F.dropout(v5, p=dropout_p, training=True)
        v7 = torch.matmul(v6, value)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 30)
key = torch.randn(4, 30)
value = torch.randn(4, 20)
dropout_p = 0.8
