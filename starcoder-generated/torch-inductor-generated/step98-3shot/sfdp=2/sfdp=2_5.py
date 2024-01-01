
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, num_heads, inv_scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
 
    def scaled_dot_product_attention(query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, query, key, value):
        return self.scaled_dot_product_attention(query, key, value, 1 / 2**.5, self.dropout_p)

# Initializing the model
m = Model(query_size=256, key_size=256, value_size=1024, num_heads=4, inv_scale_factor=1 / 2**.5, dropout_p=0)

# Inputs to the model
query = torch.randn(1, 4, 256)
key = torch.randn(1, 4, 256)
value = torch.randn(1, 4, 1024)
