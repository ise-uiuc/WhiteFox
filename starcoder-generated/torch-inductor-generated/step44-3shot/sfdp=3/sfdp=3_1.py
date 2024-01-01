
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.dropout_prob = 0.1
 
    def forward(self, q, v):
        scale_factor = 1 / (self.num_heads ** 0.5)
        # q.size == [batch_size, query_len, num_heads, seq_len]
        q = torch.einsum('bhlk,bhlm->bhlkm', q, q)
        q *= scale_factor
        attn = torch.softmax(q, dim = -1)
        attn = torch.nn.functional.dropout(attn, p=0.1)
        return torch.einsum('bhlkm,bkm->bhlk', attn, v)

# Initializing the model
n = Model()

# Inputs to the model
q = torch.randn(1, 1, 4, 6)
v = torch.randn(1, 4, 6)
