
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor):
        self.query = query
        self.key = key
        self.value = value
        self.inv_scale_factor = inv_scale_factor

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
inv_scale_factor = 8
m = Model(query, key, value, inv_scale_factor)

# Inputs to the model
dropout_p = 0.5
