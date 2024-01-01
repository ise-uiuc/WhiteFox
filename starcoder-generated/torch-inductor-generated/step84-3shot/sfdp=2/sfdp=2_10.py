
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query_tensor = torch.randn(1, 16, 5, 5)
key_tensor = torch.randn(1, 32, 7, 7) 
value_tensor = torch.randn(1, 32, 5, 5)
inv_scale_factor = 0.5
dropout_p = 0.7
m = Model()

# Inputs to the model
