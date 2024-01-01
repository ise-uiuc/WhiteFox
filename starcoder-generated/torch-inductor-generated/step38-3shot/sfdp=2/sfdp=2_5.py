
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def self_attn(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 128, 10)
key = torch.randn(1, 128, 20)
value = torch.randn(1, 128, 20)
__inv_scale_factor__ = 0.012
__dropout_p__ = 0.1
