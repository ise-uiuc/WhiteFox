
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output 

# Initializing the model
m = Model()
 
# Input to the model
query = torch.randn(12, 3, query_len, value_len)
key = torch.randn(12, 3, key_len, value_len)
value = torch.randn(12, 3, value_len, value_len)
inv_scale_factor = torch.randn(1).item()
out = m(query, key, value, inv_scale_factor)

