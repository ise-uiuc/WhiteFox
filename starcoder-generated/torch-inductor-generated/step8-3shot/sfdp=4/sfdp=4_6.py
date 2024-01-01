
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        s = s + attn_mask
        attn_weight = torch.softmax(s, -1)
        output = attn_weight @ v
        return output
 
class Model(torch.nn.Module):
    def __init__(self, num_heads: int, d_k: int, d_v: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = num_heads * d_k
        self.q_linear = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.k_linear = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.v_linear = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.attention = ScaledDotProductAttention(d_k=self.d_k)
 
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch = q.size(0)
        q = self.q_linear(q).view(batch, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(batch, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(batch, -1, self.num_heads, self.d_v).transpose(1,2)
        output = self.attention(q=q, k=k, v=v, attn_mask=attn_mask)
        output = output.transpose(1,2).contiguous().view(batch, -1, self.num_heads*self.d_v)
        return output

