
class Model(torch.nn.Module):
    def __init__(self, key_dim: int, scale_factor: float, dropout_p: float):
        super().__init__()

        self.key_dim = key_dim
        self.scale_factor = math.sqrt(key_dim)
        self.dropout_p = dropout_p

        self.q_proj = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.k_proj = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.v_proj = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights * self.scale_factor
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn = self.dropout(attn_weights)

        output = torch.matmul(attn, v)
        return output

# Initializing the model
key_dim = 17
scale_factor = 0.33
dropout_p = 0.1
m = Model(key_dim=key_dim, scale_factor=scale_factor, dropout_p=dropout_p)

# Inputs to the model
query = torch.randn(5, 7, key_dim)
value = torch.randn(5, 7, key_dim)
key = torch.randn(5, 12, key_dim)
