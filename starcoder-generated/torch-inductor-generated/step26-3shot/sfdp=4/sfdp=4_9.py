
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, nhead, d_model, dropout=0.0, bias=True):
        super().__init__()
        self.nhead = nhead
        self.dim = d_model
        self.dropout = dropout
        self.head_dim = d_model // nhead

        self.q_linear = torch.nn.Linear(self.dim, self.dim, bias=bias)
        self.k_linear = torch.nn.Linear(self.dim, self.dim, bias=bias)
        self.v_linear = torch.nn.Linear(self.dim, self.dim, bias=bias)
        self.out_linear = torch.nn.Linear(self.dim, self.dim)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # reshape q k v
        q = self.q_linear(q).view(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.nhead, self.head_dim).transpose(1, 2)

        # compute attention using function in PyTorch
        if mask is not None:
            attn_mask = (mask == 0).view(bs, 1, 1, -1).repeat(1, self.nhead, self.dim // self.nhead, 1)
            attn_mask = torch.where(attn_mask, torch.ones_like(attn_mask) * float('-inf'), attn_mask)
            q = torch.where(attn_mask.bool(), torch.zeros_like(q), q)
            k = torch.where(attn_mask.bool(), torch.zeros_like(k), k)
        scale = 1.0 / math.sqrt(self.dim // self.nhead)
        qk = torch.matmul(q, k.transpose(2, 3)) * scale
        attn_weight = torch.softmax(qk, dim=3)
        if self.dropout > 0:
            attn_weight = F.dropout(attn_weight, p=self.dropout, training=self.training)
        output = torch.matmul(attn_weight, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.dim)
        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        output = output * self.dim ** -0.5
        output = self.out_linear(output)

        # return output
        return output

# Initializing the model
m1 = MultiHeadAttention(2, 64)
m2 = MultiHeadAttention(2, 64)

# Inputs to the model
x1 = torch.randn(1, 10, 64)
x2 = torch.randn(1, 16, 64)
x3 = torch.randn(1, 16, 64)
mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  
