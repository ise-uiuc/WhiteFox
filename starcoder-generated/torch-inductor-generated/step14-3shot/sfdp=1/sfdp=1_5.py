
class Model(torch.nn.Module):
    def __init__(self, num_heads, dim_key, dim_value):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim_key)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_dropout = torch.nn.Dropout(0.1)
        self.attn_layer = torch.nn.Linear(dim_key, dim_value)
        self.norm2 = torch.nn.LayerNorm(dim_key)
        self.mlp1 = torch.nn.Linear(dim_key, dim_key)
        self.mlp_dropout1 = torch.nn.Dropout(0.1)
        self.mlp2 = torch.nn.Linear(dim_key, dim_key)
        self.mlp_dropout2 = torch.nn.Dropout(0.1)
        self.apply(self._init_weights)
 
    def forward(self, x1, x2):
        x1_ = self.norm1(x1)
        x2_ = self.norm1(x2)
        query, _ = torch.chunk(x1_, 2, dim=-1)    # split along the last dimension
        key, _ = torch.chunk(x2_, 2, dim=-1)      # split along the last dimension
        qk = torch.matmul(query, key.transpose(-2, -1))       # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(10)                   # Scale the dot product by 1/10
        softmax_qk = self.softmax(scaled_qk)    # Apply softmax to the scaled dot product
        dropout_qk = self.attn_dropout(softmax_qk)   # Apply dropout to the softmax output
        attn = self.attn_layer(dropout_qk)       # Compute the dot product of the dropout output and the value tensor
        x1_new = torch.cat([x1, attn], dim=-1)                    # Concatenate the tensor x1 and the tensor attn on the last dimension
        x1_new = x1_new + x1_                                # Add the tensor x1 and the tensor attn
        mlp = nn.functional.gelu(x1_new)                         # Apply GELU activation to the tensor x1_new
        x2_ = self.norm2(mlp)                           # Layer normalization
        mlp = x2_ + x2                                  # Add the tensor x2 and the tensor x2
        mlp = self.mlp1(x2_)                        # Project the tensor x2_ down to the dimension of dim_key
        mlp = self.mlp_dropout1(mlp)       # Apply dropout to the tensor x2_
        mlp = self.mlp2(mlp)                      # Project the tensor mlp back to the dimension of dim_key
        mlp = mlp + x2_                                # Add the tensor x2_ and the tensor mlp
        return mlp
 
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
 
        if isinstance(module, nn.Linear):
            module.bias.data.zero_()
 
# Initializing the model
m = Model(num_heads=8, dim_key=64, dim_value=32)
 
# Inputs to the model
x1 = torch.randn(1, 64, 2)        # Input tensor x1 for the query
x2 = torch.randn(1, 64, 4)        # Input tensor x2 for the key
