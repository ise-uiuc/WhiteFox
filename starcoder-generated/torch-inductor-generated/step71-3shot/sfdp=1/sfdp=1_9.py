
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
d_head = 16 # Dimension of head
n_head = 4 # Number of heads
n_key = 64 # Dimension of the key
n_value = 64 # Dimension of the value
scale_factor = 1 / (d_model ** 0.5)
inv_scale_factor = 1 / scale_factor
dropout_p = 0.1 # Dropout probability
m = Model()

# Inputs to the model
query = torch.randn(1, n_head, d_head, 256)
key = torch.randn(1, n_head, 256, n_key)
value = torch.randn(1, n_head, 256, n_value)
