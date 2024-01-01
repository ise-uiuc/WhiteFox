
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.head_dims = d_model // num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dims))
        
        self.fc_q = torch.nn.Linear(d_model, d_model, bias=True)
        self.fc_k = torch.nn.Linear(d_model, d_model, bias=True)
        self.fc_v = torch.nn.Linear(d_model, d_model, bias=True)
        self.fc_o = torch.nn.Linear(d_model, d_model, bias=True)
  
        self.dropout = torch.nn.Dropout(p=dropout)
            
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        
        query = query.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1,2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1,2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dims).transpose(1,2)
         
        attn = torch.matmul(query, key.transpose(-2,-1))
        attn = attn / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, value)
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc_o(out)
        return out    

# Initializing the model
model = MultiHeadAttention(8, 512)

# Inputs to the model
query = torch.randn(20, 8, 512)
key = torch.randn(20, 8, 512)
value = torch.randn(20, 8, 512)
