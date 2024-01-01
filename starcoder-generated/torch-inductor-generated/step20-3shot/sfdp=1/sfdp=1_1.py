
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
 
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        hidden_dim = embedding_dim * num_heads
 
        self.query_proj = torch.nn.Linear(embedding_dim, hidden_dim)
        self.key_proj = torch.nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = torch.nn.Linear(embedding_dim, hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, embedding_dim)
 
    def forward(self, query, key, value, padding_mask):
        B, T, C = query.shape
        H = self.num_heads
 
# Project inputs to the correct shapes
        query = self.query_proj(query).view(B, T, H, C)
        key = self.key_proj(key).view(B, T, H, C)
        value = self.value_proj(value).view(B, T, H, C)
        padding_mask = padding_mask.view(B, 1, T, 1)

# Add dimensions to broadcast multiplication
        query = query.view(B, T, H, C, 1)
        key = key.view(B, T, H, 1, C)
        value = value.view(B, T, H, 1, C)
        padding_mask = padding_mask.view(B, 1, 1, T, T)

# Compute the dot product of the query and key tensors
        qk = query * key

# Compute the dot product of the query and key tensors
        inv_scale_factor = math.sqrt(C)
        scaled_qk = qk / inv_scale_factor

# Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=2)

# Apply dropout to the softmax output
        dropout_p = 0
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)

# Compute the dot product of the dropout output and the value tensor
        output = dropout_qk * value

# Combine the output from different heads
        output = output.transpose(1, 2)
        output_shape = (B, H, T, C)
        output = output.reshape(*output_shape)

# Apply the final linear layer
        output = self.output_proj(output)
 
        return output

# Initializing the model
m = Model(128, 8)

# Inputs to the model
x1 = torch.randn(3, 4, 128)
x2 = torch.randn(3, 4, 128)
x3 = torch.randn(3, 4, 128)
x4 = torch.randint(2, (3, 1, 4))
