
class Model(torch.nn.Module):
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch_size, num_heads, 8, 8)
key = torch.randn(batch_size, num_heads, 8, 8)
value = torch.randn(batch_size, num_heads, 8, 8)
__inv_scale_factor__ = torch.rand(num_heads, 1, 1).fill_(512.)
__dropout_p__ = 0.0
__output__1 = m(query, key, value, __inv_scale_factor__, __dropout_p__)

