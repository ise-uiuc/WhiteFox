
class AttentionBlock(torch.nn.Module):
       def __init__(self, dim, num_heads=1, dropout_p=0.1):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // self.num_heads
            self.scale_factor = self.head_dim ** -0.5
            self.linears = torch.nn.ModuleList([
                            torch.nn.Linear(in_features=dim, out_features=dim),
                            torch.nn.Linear(in_features=dim, out_features=dim),
                            ])
            self.dropout = torch.nn.Dropout(p=dropout_p)
 
        def forward(self, query, key, value):
            # Set batch size to 1 to get an example of self-attention
            new_shape = (32, 64 * 64)
            query, key, value =  (
                            query.view(new_shape).unsqueeze(0),
                            key.view(new_shape).unsqueeze(0),
                            value.view(new_shape).unsqueeze(0))
 
            q, k, v = (
                    self.linears[0](query),
                    self.linears[1](key),
                    self.linears[2](value))
 
            query_layer = torch.nn.functional.dropout(
                        q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim),
                        p=self.dropout_p)
            key_layer = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
            value_layer = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim)
 
            q, k, v = (query_layer, key_layer, value_layer)
 
            # Compute dot product
            scaled_qk = torch.matmul(
                        q.transpose(-2, -1), k * self.scale_factor).squeeze(-2)
            softmax_qk = torch.softmax(scaled_qk, dim=-1)
            dropout_qk = self.dropout(softmax_qk)
 
            # Matrix multiplication
            attended_output = torch.matmul(dropout_qk, v)
 
            # Reshape to original shape of value
            attended_output = attended_output.squeeze(1)
            new_shape = (new_shape[1], -1)
            attended_output = attended_output.view(new_shape)
            return attended_output

# Initializing the model
ab = AttentionBlock(128)

# Inputs to the model
query = torch.randn(1, len(AB_INPUT_FEATURES), 128)
key = torch.randn(1, len(AB_INPUT_FEATURES), 128)
value = torch.randn(1, len(AB_INPUT_FEATURES), 128)
