
class Attention(torch.nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        # Create weight parameters
        self.query = torch.nn.Parameter(torch.empty(embedding_dim, num_heads))
        self.key = torch.nn.Parameter(torch.empty(embedding_dim, num_heads))
        self.value = torch.nn.Parameter(torch.empty(embedding_dim, num_heads))
        # If you choose kaiming_normal_ for the `weight_init`, you can leave the following code unchanged
        torch.nn.init.kaiming_normal_(self.query)
        torch.nn.init.kaiming_normal_(self.key)
        torch.nn.init.kaiming_normal_(self.value)
        
        # Scale parameter
        self.scale_factor = float(torch.sqrt(math.sqrt(embedding_dim)))
        
        # Create dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        assert(key.shape[-1] == value.shape[-1])
        N = query.shape[0]
        q = torch.matmul(query, self.query)
        k = torch.matmul(key, self.key)
        v = torch.matmul(value, self.value)

        # Scale
        q = q / self.scale_factor
        k = k / self.scale_factor

        # QK MatMul
        qk = torch.matmul(q, k.transpose(-2, -1))
        
        if mask is not None:
            qk = qk.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e8)
        
        # Softmax and dropout
        qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = self.dropout_layer(qk)

        # Output
        qkv = torch.matmul(qk, v).reshape(N, -1, self.num_heads * self.embedding_dim)
        return qkv

# Initializing the model
m = Attention(num_heads=8, embedding_dim=32)

# Inputs to the model
x1 = torch.randn(6, 196, 32)
x2 = torch.randn(6, 196, 32)
x3 = torch.randn(6, 196, 32)
mask = torch.randn(6, 196)
