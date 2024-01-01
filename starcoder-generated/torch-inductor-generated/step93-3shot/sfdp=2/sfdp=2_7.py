
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, dropout_p):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.dropout_p = dropout_p

    def forward(self, query, key, value, residual):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(key.size(-1))
        scaled_qk = qk.mul(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p, training=True)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_attention_heads=8, dropout_p=0.1)

# Inputs to the model
query = torch.randn(1, 16, 512) # The query must have the dimension [B, L_query, D], where L_query is the sequence length, D is the feature dimension
key = torch.randn(1, 128, 512)
value = torch.randn(1, 128, 512)
residual = torch.randn(1, 16, 512)
