
class Model(torch.nn.Module):
    def __init__(self,
                 d_query,
                 d_key,
                 d_value,
                 n_heads,
                 n_hiddens):
        super().__init__()
        self.n_heads = n_heads
        self.d_value = d_value
 
        self.w_q = torch.nn.Linear(d_query, n_heads * d_value)
        self.w_k = torch.nn.Linear(d_key, n_heads * d_value)
        self.w_v = torch.nn.Linear(d_value, n_heads * d_value)
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.linear = torch.nn.Linear(n_heads * d_value, n_hiddens)
 
 
    def forward(self, query, key, value, mask=None):
        bs = query.size(0)
        scale_factor = 1.0 / math.sqrt(self.d_value)
        q = self.w_q(query).view(bs, -1, self.n_heads, self.d_value)
        k = self.w_k(key).view(bs, -1, self.n_heads, self.d_value)
        v = self.w_v(value).view(bs, -1, self.n_heads, self.d_value)
 
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, *q.size()[2:])
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, *k.size()[2:])
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, *v.size()[2:])
 
        if mask:
            scale_factor = scale_factor.repeat(bs * self.n_heads).view(-1, 1, 1)
            scaled_attn_logits = self.scaled_dot_product_attention(q, k, v, mask, scale_factor)
        else:
            scaled_attn_logits = self.scaled_dot_product_attention(q, k, v, scale_factor)
 
        scaled_attn_logits = scaled_attn_logits.view(self.n_heads, bs, -1, *scaled_attn_logits.size()[1:]).permute(1, 2, 0, 3, 4).contiguous()
        output = self.linear(scaled_attn_logits.view(bs, -1, self.n_heads * self.d_value))
 
        return output

# Initializing the model
m = Model(d_query, d_key, d_value, n_heads, n_hiddens)

# Inputs to the model
query = torch.randn(2, 3, 4)
key = torch.randn(2, 5, 4)
