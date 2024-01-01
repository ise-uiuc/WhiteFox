
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor=1.0, dropout_p=0.0, num_attention_heads=1):
        super().__init__()
        assert num_attention_heads > 0
        self.qk_matmul = torch.nn.MultiheadAttention(embed_dim, num_attention_heads).qk_proj_weight.T
        self.v_matmul = torch.nn.MultiheadAttention(embed_dim, num_attention_heads).v_proj_weight.T

    def forward(self, query, key, value):
        qk = torch.matmul(query, self.qk_matmul) # Compute the dot product of the query and key tensors
        scaled_qk = qk / self.inv_scale_factor # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = torch.matmul(dropout_qk, value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, num_attention_heads, query_len, query_len)
key = torch.randn(1, num_attention_heads, key_len, query_len)
value = torch.randn(1, num_attention_heads, value_len, query_len)
