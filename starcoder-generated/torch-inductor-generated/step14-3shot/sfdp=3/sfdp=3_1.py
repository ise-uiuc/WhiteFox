
class M(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = M()

# Inputs to the model
query = torch.randn(1, 3, 32, 32)
key = torch.randn(1, 16, 32, 32)
value = torch.randn(1, 16, 32, 32)
scale_factor = torch.tensor(1.0 / math.sqrt(16))
dropout_p =.3
