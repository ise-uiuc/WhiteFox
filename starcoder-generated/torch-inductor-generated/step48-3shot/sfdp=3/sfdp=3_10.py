
class Model(torch.nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout_p=0.1):
        super(Model, self).__init__()
        self.w_q = torch.nn.Linear(d_model, n_head * d_k)
        self.w_k = torch.nn.Linear(d_model, n_head * d_k)
        self.w_v = torch.nn.Linear(d_model, n_head * d_v)
        self.w_o = torch.nn.Linear(n_head * d_v, d_model)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def attention(self, q, k, v, mask=None):
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        d_model = q.size(-1)
        assert d_model == k.size(-1) == v.size(-1)
        residual = q
        q = self.w_q(q).view(n_head, -1, d_k).transpose(1, 2) # (n*b) x lq x dk
        k = self.w_k(k).view(n_head, -1, d_k).transpose(1, 2) # (n*b) x lk x dk
        v = self.w_v(v).view(n_head, -1, d_v).transpose(1, 2) # (n*b) x lv x dv
        scaled_attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k).item() # (n*b) x lq x lk
        if mask is not None:
            assert mask.dim() == 2
            mask = mask.unsqueeze(1).expand(mask.size(0), n_head, mask.size(1)) # (n*b) x.. x.. -> (n*b) x n x lq x lk
            scaled_attention = scaled_attention.masked_fill_(mask == 0, -1e-9)
        attention = self.dropout(torch.nn.functional.softmax(scaled_attention, dim=-1)) # (n*b) x lq x lk
        output = torch.matmul(attention, v) # (n*b) x lq x dv
        output = output.transpose(1, 2).contiguous().view(n_head, -1, d_model) # n*b x lv x lq
        output = self.w_o(output)
        return output + residual
 
    def forward(self, q, k, v, mask=None):
        d_model = q.size(-1)
        # reshape qkv input
        n = q.size(0)
        bsz = q.size(1)
        q = q.view(n, bsz, -1)
        k = k.view(n, bsz, -1)
        v = v.view(n, bsz, -1)
        # reshape ouput
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # apply attention
        y = self.attention(q, k, v, mask)
        y = y.view(n, bsz, -1)
        y = self.dropout(y)
        # apply fc layer
        # reshape qkv output
        # reshape y back
        return y

# Initializing the model
m = Model(d_model=64, dropout_p=0.1)

# Inputs to the model
q = torch.randn(1, 8, 64)
k = torch.randn(1, 4, 64)
v = torch.randn(1, 4, 64)
mask = torch.ByteTensor([1, 0, 0, 1])
