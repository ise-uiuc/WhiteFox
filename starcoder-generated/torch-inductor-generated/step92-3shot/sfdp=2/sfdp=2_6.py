
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model.
m = Model()

# Inputs to the model.
key = torch.randn(4, 3, 3136)
query = torch.randn(4, 3, 3136)
value = torch.randn(4, 3, 3136)
dropout_p = 0.1
inv_scale_factor = 0.0625
