
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(query.size(-1))
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk,
                                                 p=0.4)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 10, 10)
key = torch.randn(1, 1, 10, 10)
value = torch.randn(1, 1, 10, 10)
