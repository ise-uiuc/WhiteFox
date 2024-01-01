
class Model(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(dim)
        self.dropout_p = 0.2
        self.heads = heads
 
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        zscore = qk.mul(self.scale_factor)
        attn_weights = zscore.softmax(dim = -1)
        dout = self.dropout_p
        dropout_attn_weights = torch.nn.functional.dropout(attn_weights, p=dout)
        output = dropout_attn_weights.matmul(value)
        return output
 

# Initializing the model
input_dim = value.size(-1)
multi_headed_attn = Model(input_dim, num_heads)

# Input to the model
v1 = torch.randn(1, 5, input_dim)
v2 = torch.randn(1, 4, input_dim)
v3 = torch.randn(1, 6, input_dim)
