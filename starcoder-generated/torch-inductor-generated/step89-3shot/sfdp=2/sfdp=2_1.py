
class Model(torch.nn.Module):
    def __init__(self, query_head, key_head, value_head):
        super().__init__()
        self.query = torch.nn.Linear(query_head, query_head)
        self.key = torch.nn.Linear(key_head, key_head)
        self.value = torch.nn.Linear(value_head, value_head)
        self.dropout = torch.nn.Dropout(p)
 
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = self.dropout(torch.matmul(q, k.t()) / math.sqrt(h))
        x = torch.matmul(x, v)
        return x

# Initializing the model
m = Model(query_head, key_head, value_head)

# Input to the model
x = torch.randn(1, h, s)
