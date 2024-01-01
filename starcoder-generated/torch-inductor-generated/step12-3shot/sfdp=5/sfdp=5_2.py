s
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.h = h
        self.w_qs = nn.Linear(d_model, h * d_k)
        self.w_ks = nn.Linear(d_model, h * d_k)
        self.w_vs = nn.Linear(d_model, h * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        
        self.fc1 = nn.Linear(h * d_v, d_model)
        nn.init.xavier_normal_(self.fc1.weight)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, qs, ks, vs, attn_mask=None):
        len_q = qs.size(1)
        len_k = ks.size(1)
        residual = qs
        q = qs.view(-1, len_q, self.h, self.d_k)
        k = ks.view(-1, len_k, self.h, self.d_k)
        v = vs.view(-1, len_k, self.h, self.d_v)
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, len_q, self.d_k) # (N*h, len_q, d_k)
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, len_k, self.d_k) # (N*h, len_k, d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, len_k, self.d_v) # (N*h, len_k, d_v)
        mask = attn_mask.repeat(self.h, 1, 1) # (N*h, len_q, len_k)
        
        output, attn = self.attention(q, k, v, mask)
        
        output = output.view(-1, self.h, len_q, self.d_v)
        output = output.permute(0, 2, 1, 3).contiguous().view(-1, len_q, self.h * self.d_v) # (N, len_q, h*d_v)
        
        attn = attn.view(-1, self.h, len_q, len_k)
        
        output = self.dropout(self.fc1(output))
        return self.layer_norm(output + residual), attn
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        return self.layer_norm(output + residual)

# Positional Encoding
def get_sinusoid_encoding_table(n_position, d_hid):
    