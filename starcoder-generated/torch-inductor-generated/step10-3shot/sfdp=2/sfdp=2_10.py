
class Model(torch.nn.Module):
    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 50, 32, 64)
key = torch.randn(1, 50, 32, 64)
value = torch.randn(1, 50, 32, 64)
dropout_p = 0.5
inv_scale_factor = 0.5
