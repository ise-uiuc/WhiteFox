
class Model(torch.nn.Module):
    def __init__(self, dim, nb_head, dropout_p):
        super().__init__()
        self.dim = dim
        self.nb_head = nb_head
        self.dropout_p = dropout_p
        self.head_dim = dim // nb_head
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
 
    def forward(self, xq, xk, xv, attn_mask):
        # query, key, value
        q = self.query(qv)
        k = self.key(kv)
        v = self.value(xv)
        # compute the scaled dot product
        qk = q @ k.transpose(-2, -1)
        # normalize by the query/key dimension and the square-root of the key/query dimension
        qk = qk / math.sqrt(q.size(-1))
        # add attention mask
        v = v + attn_mask
        # apply softmax
        attn_weight = torch.softmax(qk, dim=-1)
        # apply dropout
        v =  torch.dropout(attn_weight, self.dropout_p, True)
        # compute the dot product
        output = torch.matmul(k, v)
        return output

# Initializing the model
m = Model(2500, 5, 0.5)

# Inputs to the model
xq = torch.randn(16, 37, 2500)
k = torch.randn(48, 37, 2500)
v = torch.randn(48, 37, 2500)
attn_mask = torch.randn(16, 37, 48, requires_grad=False)
