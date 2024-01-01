
class LinearAttention(torch.nn.Module):
    def __init__(self, input_dim: int, num_heads: int,
                 scale_factor: float = 1.0, dropout_p: float = 0.5):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
        self.q_proj = torch.nn.Linear(input_dim, num_heads, bias=False)
        self.k_proj = torch.nn.Linear(input_dim, num_heads, bias=False)
        self.v_proj = torch.nn.Linear(input_dim, num_heads, bias=False)
        self.o_proj = torch.nn.Linear(input_dim, num_heads, bias=False)
        self.dropout1 = torch.nn.Dropout(dropout_p)
        self.dropout2 = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, mask=None):
        q = self.q_proj(query).reshape(*query.shape[0:2], -1)
        k = self.k_proj(key).reshape(*key.shape[0:2], -1)
        v = self.v_proj(value).reshape(*value.shape[0:2], -1)
        q /= math.sqrt(q.shape[-1])
        
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        if mask is not None:
            qk += mask
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = self.dropout1(softmax_qk)
        output = self.o_proj(dropout_qk.matmul(v).reshape(*query.shape[0:2], -1))
        if mask is not None:
            output += mask
        output = self.dropout2(output)
        return output

# Initializing the model
n = LinearAttention(input_dim=256, num_heads=2, dropout_p=0.5)

# Inputs to the model
x1 = torch.randn(1, 2, 4, 256)
x2 = torch.randn(1, 2, 8, 256)
x3 = torch.randn(1, 2, 8, 256)
mask = torch.randn_like(x1, dtype=torch.bool)
