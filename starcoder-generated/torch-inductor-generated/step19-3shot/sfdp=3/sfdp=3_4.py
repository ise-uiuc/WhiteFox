
class MultiheadScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads!= 0 or num_heads <= 0:
            raise ValueError(
                f"The feature dimension {dim} should be divisible by the number of heads {num_heads}."
            )
 
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
 
        self.queries = torch.nn.Linear(dim, dim)
        self.keys = torch.nn.Linear(dim, dim)
        self.values = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value, mask=None, dropout_p=0.1):
        batch_size = query.size(0)

        query_proj = self.queries(query)
        key_proj = self.keys(key)
        value_proj = self.values(value)

        # Split the tensors with batch dimension into pairs with feature dimension to form q, k, and v.
        query_pairs = query_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_pairs = key_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_pairs = value_proj.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
 
        q = query_pairs.query
        k = key_pairs.key
        v = value_pairs.value

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        scaled_attn = attn.softmax(dim=-1)
        attn_drop = torch.nn.functional.dropout(scaled_attn, p=dropout_p)
        out = attn_drop @ v
 
        out_pairs = torch.cat([out_i.unsqueeze(2) for out_i in torch.split(out, batch_size, dim=0)], dim=2)
        out = out_pairs.out
        return out

transformer = torch.nn.Transformer(num_encoder_layers=5, num_decoder_layers=5)

# Inputs to the model
x1 = torch.randn(20, 32, 512)
x2 = torch.randn(20, 32, 512)
mask = torch.randn(20, 1, 32, 32).to(torch.bool)
dropout = 0.5
