
class Model(torch.nn.Module):
    def forward():
        qk = torch.matmul(query, key.transpose(-1, -2))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 64, 128)
key = torch.randn(8, 128, 64)
value = torch.randn(8, 64, 64)
