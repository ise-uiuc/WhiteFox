
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 4
        self.dk = 64
        self.dropout_p = 0.1
 
    def forward(self, q, k, v, attn_mask):
        q = q / math.sqrt(self.dk)
        dots = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0., float('-inf')).masked_fill(attn_mask == 1., float(0.0))
        dots = dots + attn_mask
        attn = torch.softmax(dots, dim=-1)
        attn = torch.dropout(attn, self.dropout_p, True)
        out = torch.matmul(attn, v)
        return out

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(16, 32, 64)
k = torch.randn(16, 32, 64)
v = torch.randn(16, 32, 64)
attn_mask = torch.rand(q.size(0), q.size(1), k.size(1)) < 0.5
