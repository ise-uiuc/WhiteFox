
embed_dim = 4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.query = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        k = self.key(key)
        v = self.value(value)
        q = self.query(query)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, embed_dim)  
key = torch.randn(1, embed_dim)
value = torch.randn(1, embed_dim)
inv_scale_factor = 1e14
dropout_p = 0.0
