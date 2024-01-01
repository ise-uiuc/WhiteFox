
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model for computing attention
query = torch.randn(1, 8, 64)
key = torch.randn(1, 6, 64)
value = torch.randn(1, 6, 64)
scale_factor = torch.tensor(4.0, dtype=torch.float32)
dropout_p = torch.tensor(0.5, dtype=torch.float64)
