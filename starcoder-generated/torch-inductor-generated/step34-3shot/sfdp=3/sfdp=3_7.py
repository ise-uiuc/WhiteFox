
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_heads, d_model)

# Inputs to the model
query = torch.randn(B, N, M)
key = torch.randn(B, S, M)
value = torch.randn(B, S, M)
scale_factor = 10 # for example
dropout_p = 0.5
