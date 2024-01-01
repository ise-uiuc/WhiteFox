
class Model(torch.nn.Module):
    def __init__(self, hidden_size: int = 128, n_heads: int = 8, drop_out: float = 0.1, use_mask: bool = False):
        super().__init__()
        self.use_mask = use_mask
        self.hidden_size = hidden_size
        self.num_heads = n_heads
        self.projection = torch.nn.Linear(hidden_size, hidden_size * 3)
 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        q = self.projection(q)
        q, k, v = map(
            lambda x: x.reshape(x.size(0), x.size(1), self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2),
            (q, k, v))
        q, k, v = q * (self.hidden_size // self.num_heads) ** -0.5, k, v
        qk = q.matmul(k.transpose(2, 3))
        if attn_mask is not None and self.use_mask:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_mask = (1.0 - attn_mask) * float('-inf')
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, p=drop_out, training=self.training)
        output = attn_weight.matmul(v)
        output = output.transpose(1, 2).reshape(output.size(0), output.size(1), -1)
        return output

# Initializing the model
m = Model(hidden_size=64, n_heads=8)
# Inputs to the model
q = torch.randn(1, 64, 32, 64)
k = torch.randn(1, 64, 32, 64)
v = torch.randn(1, 64, 32, 64)
