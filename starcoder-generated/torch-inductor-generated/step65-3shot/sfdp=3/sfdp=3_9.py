
class TransformerAttention(torch.nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, scale=False):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.scale = scale
 
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)
 
    def forward(self, q, k, v, attn_mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
 
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.nhead, 1, 1)
 
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask, self.nhead, self.d_model, self.scale)
        attn_out = attn_out.transpose(0, 1).contiguous().view(q.size(0), -1)
        out = self.fc(attn_out)
        return out

# Initializing the model
m = TransformerAttention()

# Inputs to the model
q = torch.randn(4, 16, 512)
k = torch.randn(4, 16, 512)
v = torch.randn(4, 16, 512)
attn_mask = torch.randn(4, 8, 16)
