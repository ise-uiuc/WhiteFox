
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p=0.0, scale_value=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_value = scale_value
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout_p)
        self.output = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, inputs, att_mask):
        # inputs: (b, l, d)
        q, k, v = [i.contiguous() for i in torch.chunk(inputs, 3, dim=-1)]
        # q, k, v: (b, l, h)
        k_t = k.transpose(-2, -1)
        # k_t: (b, h, l)
        # attn: (b, l, l)
        attn = torch.matmul(q, k_t)
        inv_scale_factor = max(mask.squeeze().float().mean(), 1e-12)
        # attn: (b, l, l)
        attn.mul_(self.scale_value / inv_scale_factor)
        attn = self.softmax(attn) * att_mask.unsqueeze(1)
        attn = self.dropout(attn)
        # attn: (b, l, d)
        output = torch.matmul(attn, v)
        return output

# Initializing the model
m = ScaledDotProductAttention(hidden_size=64)

# Inputs to the model
mask = torch.zeros([1, 64, 64]).scatter_(2, torch.tensor([[0,]]*64).unsqueeze(0), 1)
query = value = torch.randn(1, 64, 64)
