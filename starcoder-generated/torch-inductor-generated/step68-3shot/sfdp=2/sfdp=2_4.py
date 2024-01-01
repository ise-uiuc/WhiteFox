
class Model(nn.Module):
    def __init__(self, num_queries, embed_dim):
        super(Model, self).__init__()
        num_heads = 128
        attn_head_size = embed_dim // num_heads
 
        # Query
        self.query = nn.Linear(embed_dim, embed_dim)
 
        # Key
        self.key = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for i in range(num_heads)
        ])
 
        # Value
        self.value = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for i in range(num_heads) 
        ])
 
        # Scale factor
        self.invs = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0), requires_grad=True)
            for i in range(num_heads)
        ])
 
        # Dropout probability
        self.drop = nn.Dropout(0.5)
 
    def forward(self, query, key, value):
        # Shape of query and key: (batch_size, num_queries, embed_dim)
        # Shape of value: (batch_size, num_values, embed_dim)
        # Shape of inv: (num_heads,)
        q = self.query(query).unsqueeze(1)
        key = torch.cat(
            [key.unsqueeze(1) for i in range(len(self.key))],
            dim=1
        )
        value = torch.cat(
            [value.unsqueeze(1) for i in range(len(self.value))],
            dim=1
        )
        inv_scale = torch.cat(
            [self.invs[i] for i in range(len(self.invs))],
            dim=0
        )
 
        # Shape of q: (batch_size, 1, num_queries, embed_dim)
        # Shape of key: (batch_size, num_heads, num_values, embed_dim)
        # Shape of value: (batch_size, num_heads, num_values, embed_dim)
        # Shape of inv: (num_heads, 1)
        q = q.transpose(1, 2)
        q = q * inv_scale.view(1, -1, 1, 1)
        q = q.transpose(1, 2)
 
        # Shape of attn: (batch_size, num_heads, num_queries, num_values)
        attn = torch.matmul(q, key.transpose(-2, -1))
        attn /= np.sqrt(key.size(-1))
 
        # Apply dropout
        attn = self.drop(attn)
 
        # Apply softmax
        attn = attn.softmax(dim=-1)
 
        # Compute result
        # Shape of x: (batch_size, num_heads, num_queries, embed_dim)
        x = attn.matmul(value)
 
        # Concatenate over the num_heads dimension
        x = torch.cat(
            torch.split(
                x, len(self.value)
            ),
            dim=-1
        )
 
        # Shape of x: (batch_size, num_queries, embed_dim)
        return x

# Initializing the model
m = Model(100, 1024)

# Inputs to the model
query = torch.randn(50, 100, 1024)
keys = torch.randn(50, 120, 1024)
values = torch.randn(50, 120, 1024)
