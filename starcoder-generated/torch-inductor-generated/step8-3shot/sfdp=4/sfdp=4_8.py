
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
 
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
    def forward(self, q, k, v, attention_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
 
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
 
        # Transpose for attention dot product: b x n x lq x dv
        q = q.transpose(1, 2)
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
 
        # Transpose to move the head dimension back: b x lq x (n*dv)
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        attn_output = self.dropout(self.fc(attn_output))
        attn_output = self.layer_norm(residual + attn_output)
        return attn_output

# Initializing the model
m = Model(4, 256, 64, 64, 4, 0.1)
m.init_weights()

# Inputs to the model
x_q = torch.randn(2, 5, 256)
x_k = torch.randn(2, 8, 256)
x_v = torch.randn(2, 8, 256)
