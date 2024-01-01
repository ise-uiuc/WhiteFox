
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.W_q = torch.nn.Linear(dim, dim)
        self.W_k = torch.nn.Linear(dim, dim)
        self.W_v = torch.nn.Linear(dim, dim)
        self.W_o = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, query, key, value):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
 
        scale_factor = torch.sqrt(torch.tensor(key.size(-1)))
        q = q.div(scale_factor)
        qk = torch.matmul(q, torch.transpose(k, -2, -1))
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, v)
 
        output = self.W_o(output)
        return output

# Initializing the model
m = Model(dim=8, num_heads=8, dropout_p=0.1)

# Inputs to the model
query = torch.rand(1, 8, 64, 64)
key = torch.rand(1, 8, 64, 64)
value = torch.rand(1, 8, 64, 64)
