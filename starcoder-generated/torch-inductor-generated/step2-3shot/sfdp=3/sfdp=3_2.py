
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1. / math.sqrt(hidden_size // num_heads)
        self.dropout_layer = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, dropout_p=dropout_p):
        # Expand rank for tensors
        query = query.unsqueeze(-1).expand(-1, -1, -1, feature_size)
        key = key.expand(-1, -1, feature_size, -1)
        value = value.expand(-1, -1, -1, feature_size)
        # Compute dot product
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale dot product
        scaled_qk = qk.mul(self.scale_factor)
        # Apply softmax
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout
        dropout_qk = self.dropout_layer(softmax_qk)
        # Compute the output
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Attention()

# Inputs to the model
query = torch.randn(1, 1, hidden_size // num_heads)
key = torch.randn(1, 1, hidden_size // num_heads)
value = torch.randn(1, 1, hidden_size // num_heads)
