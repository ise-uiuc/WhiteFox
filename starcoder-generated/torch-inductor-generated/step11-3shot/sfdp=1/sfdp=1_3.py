
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, feature_dim, num_heads, inv_scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(num_heads, query_dim // num_heads, 1, 1))
        self.key = torch.nn.Parameter(torch.randn(num_heads, key_dim // num_heads, 1, 1))
        self.value = torch.nn.Parameter(torch.randn(num_heads, value_dim // num_heads, 1, 1))
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(-1)
        self.inv_scale_factor = inv_scale_factor
        self.dropout = torch.nn.Dropout(dropout_p)
    
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = self.softmax(scaled_qk) # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output
    
# Initializing the model
query_dim = 256
key_dim = 256
value_dim = 512
feature_dim = 512
num_heads = 8
inv_scale_factor = 2.0 ** 0.5
dropout_p = 0.2
m = Model(query_dim, key_dim, value_dim, feature_dim, num_heads, inv_scale_factor, dropout_p)


# Inputs to the model
query = torch.randn(1, feature_dim, 1, 1)
key = torch.randn(1, feature_dim, 1, 1)
value = torch.randn(1, feature_dim, 1, 1)
