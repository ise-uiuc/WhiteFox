
class Model(torch.nn.Module):
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing key, value, query, and inv_scale_factor
key = torch.randn(1, 32, 64, 64)
value = torch.randn(1, 32, 64, 64)
query = torch.randn(1, 32, 64, 64)
inv_scale_factor = torch.randn(1)
dropout_p = torch.zeros(1)
