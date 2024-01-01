
class Model(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
 
    def forward(self, query, key, value, d):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = d ** -0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
d = 128
m = Model(d)

# Inputs to the model
query = torch.randn(1, 128, 32)
key = torch.randn(1, 128, 64)
value = torch.randn(1, 128, 64)
