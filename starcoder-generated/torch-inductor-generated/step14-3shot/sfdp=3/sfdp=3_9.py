
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_heads
        self.d_model = args.d_model
 
        self.query = torch.nn.Linear(args.d_model, args.d_model)
        self.key = torch.nn.Linear(args.d_model, args.d_model)
        self.value = torch.nn.Linear(args.d_model, args.d_model)
 
 
 
    def forward(self, q, k, v, mask):
        bsz = q.size(0)
        query = self.query(q).view(bsz, -1, self.num_heads, self.d_model//self.num_heads).transpose(1,2)
        key = self.key(k).view(bsz, -1, self.num_heads, self.d_model//self.num_heads).transpose(1,2)
        value = self.value(v).view(bsz, -1, self.num_heads, self.d_model//self.num_heads).transpose(1,2)
 
 
 
        scaled_qk = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask, -np.inf)
        scale_factor = 1 / math.sqrt(self.d_model // self.num_heads)
        softmax_qk = scaled_qk.mul(scale_factor).softmax(dim=-1)
 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.15, training=self.training)
        output = torch.matmul(dropout_qk, value)
 
        return output.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
 
 
class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args)
 
 
 
    def forward(self, query, key, value, mask):
        return self.attn(query, key, value, mask)
    

# Initializing the model with random parameters
args = lambda: None
args.d_model = 768
args.num_heads = 12
model = Model(args)

# Initial inputs to the model
query = torch.randn(1, 14*8, 768)
key = torch.randn(1, 14*8, 768)
value = torch.randn(1, 14*8, 768)
mask = torch.ones(1, 14, 14).to(torch.bool)
