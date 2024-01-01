
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor=10000.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 16, 16)
key = torch.randn(1, 4, 16, 16)
value = torch.randn(1, 4, 16, 16)
