
class Model(Module):
    def __init__(self, query, key, value):
        super().__init__()
        self.m1 = torch.nn.Linear(query.size()[0], query.size()[2], bias=False)
        self.m2 = torch.nn.Linear(key.size()[0], key.size()[2], bias=False)
        self.m3 = torch.nn.Linear(value.size()[0], value.size()[2], bias=False)
        return
 
    def forward(self, query, key, value, scale_factor, dropout_p, seed=0):
        q = self.m1(query)
        k = self.m2(key)
        v = self.m3(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p, training=True, seed=seed)
        return torch.matmul(dropout_qk, v)

# Initializing the model
query = torch.ones(1, 3, 4)
key = torch.ones(1, 5, 4)
value = torch.ones(1, 5, 3)
torch.manual_seed(0)
m = Model(query, key, value)

# Inputs to the model
