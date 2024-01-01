
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

# Inputs for the model
query = torch.randn(3, 5)
key = torch.randn(3, 5, 7)
value = torch.randn(3, 7, 5)
dropout_p = 0.9
inv_scale_factor = float(1.0 / 7)
