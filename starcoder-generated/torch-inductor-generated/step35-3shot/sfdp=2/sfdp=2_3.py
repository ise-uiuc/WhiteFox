
class Model(torch.nn.Module):
    def forward(self, query, key, value, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1 / (dim ** 0.25)).to(query.device)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(10, 3, 64, 64)
key = torch.randn(10, 3, 64, 64)
value = torch.randn(10, 3, 64, 64)
dropout_p = 0.1
