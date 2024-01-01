
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_query = torch.nn.Linear(feature_size, feature_size)
        self.linear_key = torch.nn.Linear(feature_size, feature_size)
        self.linear_value = torch.nn.Linear(feature_size, feature_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.layernorm1 = torch.nn.LayerNorm(feature_size)
        self.layernorm2 = torch.nn.LayerNorm(feature_size)
 
    def forward(self, q, k, v, attn_mask):
        q = rearrange(self.linear_query(q), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.linear_key(k), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.linear_value(v), 'b n (h d) -> b h n d', h=h)
        attn_mask = repeat(attn_mask,'s s -> b h s s', b=b, h=h) # Repeat the attention mask
        attn = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        attn = attn + attn_mask
        attn_weight = torch.softmax(attn, dim=-1)
        attn_weight = rearrange(self.dropout(attn_weight), 'b h n s -> b n (h s)')
        output = attn_weight @ v
        output = rearrange(output, 'b n (h d) -> b n h d', h=h)
        output = self.layernorm1(output + q)
        output = rearrange(self.dropout(output), 'b n h d -> b (n h) d')
        output = self.layernorm2(output + k)
        output = rearrange(self.dropout(output), 'b (n h) d -> b n h d', n=seq_len)
        output = output + v
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, model_dim, sequence_len)
k = torch.randn(1, model_dim, sequence_len)
v = torch.randn(1, model_dim, sequence_len)
attn_mask = torch.tril(torch.ones(sequence_len, sequence_len))[None, None] == 0
model_output = m(q, k, v, attn_mask)

