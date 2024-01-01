
class Model(torch.nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.qkv = torch.nn.Linear(emb_dim, 3 * emb_dim)
        self.attn_mask = torch.zeros(20, 20)
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, x):
        k1 = self.qkv(x)
        q1, k2, v1 = k1[:, :emb_dim], k1[:, 1 * emb_dim], k1[:, 2 * emb_dim:]
        query = torch.einsum("bih,bij->bhj", q1, k2)
        key = torch.einsum("bij,bic->bjk", q1, v1)
        attn_mask = torch.zeros(20, 20)
        attention_weights = torch.softmax((query @ key.transpose(-2, -1)) / query.size(-1) + attn_mask, dim=-1)
        attention_weights = self.dropout(attention_weights)
        x1 = torch.einsum("bij,bjk->bik", attention_weights, v1)
        x2 = self.qkv(x1)
        x3 = x2[:, 0*emb_dim] + x2[:, 1*emb_dim] + x2[:, 2*emb_dim]
        return x3

# Initializing the model with hyperparameters
m = Model(16, 2)

# Inputs to the model
x = torch.rand(31, 33)
y = m(x)

