
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 1, 1, 16)
key = torch.randn(4, 1, 1, 16)
value = torch.randn(4, 1, 1, 16)
scale_factor = 0.1
dropout_p = 0.1
