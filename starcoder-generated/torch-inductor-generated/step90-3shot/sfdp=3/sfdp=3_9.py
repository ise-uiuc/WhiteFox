
class Model(torch.nn.Module):
    def forward(i, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(query.shape[-1])
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.1
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Getting the inputs for the model
query = torch.randn(1, 2, 8, 8)
key = torch.randn(1, 2, 8, 8)
value = torch.randn(1, 2, 8, 8)
