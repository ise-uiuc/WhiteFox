
class Attention(nn.Module):
    def __init__(self, d_model, dropout_p, scale):
        super().__init__()
        assert d_model % scale == 0
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_p)
 
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
        self.scale = torch.sqrt(torch.FloatTensor([scale]))
        
    def forward(self, q, k, v, mask=None):
        batch_size, len_q, d_model = q.shape
        batch_size, len_k, d_model = k.shape
        batch_size, len_v, d_model = v.shape
 
        q = self.q(q).view(batch_size, len_q, self.h, self.d_k).permute(0, 2, 1, 3)  # (batch_size, num_heads, len_q, d_k)
        k = self.k(k).view(batch_size, len_k, self.h, self.d_k).permute(0, 2, 3, 1)  # (batch_size, num_heads, d_k, len_k)
        v = self.v(v).view(batch_size, len_v, self.h, self.d_k).permute(0, 2, 1, 3)  # (batch_size, num_heads, len_v, d_k)

        # scaled_attention: (batch_size, num_heads, len_q, d_k)
        # attention: (batch_size, num_heads, len_q, len_k)
        scaled_attention = torch.matmul(q / self.scale, k)
        attention = nn.functional.softmax(scaled_attention, dim=-1)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e4)

        # output: (batch_size, len_q, num_heads, d_k)
        output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, len_q, -1)

        return self.dropout(output)

# Initializing the model
d_model = 32
scale = 1
dropout_p = 0.6
self_attention = Attention(d_model=d_model, dropout_p=dropout_p, scale=scale)

# Inputs to the system
x1 = torch.randn(1, 10, d_model)
x2 = torch.randn(1, 20, d_model)
x3 = torch.randn(1, 20, d_model)
mask = torch.zeros(1, 10, 20)
