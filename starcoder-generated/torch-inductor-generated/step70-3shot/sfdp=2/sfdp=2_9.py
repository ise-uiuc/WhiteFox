
class Model(torch.nn.Module):
    def forward(self, input, key, value, _scale_factor, _dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / _scale_factor
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=_dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 4)
key = torch.randn(1, 4, 8)
value = torch.randn(1, 4, 8)
scale_factor = 1
dropout_p = 0.7
