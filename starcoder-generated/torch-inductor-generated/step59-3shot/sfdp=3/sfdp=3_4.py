
class Model(torch.nn.Module):
    def forward(self, query, key, value, dropout_p=0.0):
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 256)
key = torch.randn(1, 16, 256)
value = torch.randn(1, 16, 256)
