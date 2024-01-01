
class Attention(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(query_size, key_size))
        self.dropout_p = dropout_p
    
    def forward(self, query, key, value, mask=None, training=False): 
        key_dim = key.size(-1)
        inv_scale_factor = (key_dim ** -0.5)        
        query, key = self.query.expand_as(query), self.query.expand_as(key)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p, training=training)
        output = dropout_qk.matmul(value)
        return output

at = Attention(1024, 1024, 1024, 0.3)

# Inputs to the model
query = torch.randn(32, 1024, 1)
key = torch.randn(32, 1024, 100)
value = torch.randn(32, 1024, 100)
