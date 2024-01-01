
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, value, key, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)) 
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 25, 512)
value = torch.randn(1, 25, 512)
key = torch.randn(1, 25, 512)
inv_scale_factor = 1.0 / 512.0
dropout_p = 0.1
