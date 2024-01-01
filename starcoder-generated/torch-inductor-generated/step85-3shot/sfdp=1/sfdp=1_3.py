
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
  ...
 
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
query = torch.rand(1, 64, 512)
key = torch.rand(1, 64, 512)
value = torch.rand(1, 64, 512)
inv_scale_factor = 1. / math.sqrt(512)
dropout_p = 0.5
