
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = query.matmul(key.transpose(-1, -2))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, q_len, 128, 128)
key = torch.randn(1, kv_len, 64, 128)
value = torch.randn(1, kv_len, 64, 128)
inv_scale_factor = torch.randn(1, 1, 1)
