
class Model(torch.nn.Module):
    def forward(self, query, key, value, dropout_p):
        # 256 is the `n_head`
        # 4 is the `d_key`
        # 256 is the `d_value`
        qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(4)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 256, 80, 64)
key = torch.randn(1, 256, 40, 64)
value = torch.randn(1, 256, 40, 64)
dropout_p = 0.1
output = m(query, key, value, dropout_p)
