
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_tensor, key_tensor, value_tensor, scale_factor, dropout_p):
        qk = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value_tensor)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 512, 64)
key = torch.randn(1, 512, 64)
value = torch.randn(1, 512, 64)
scale_factor = torch.scalar_tensor(28.22)
dropout_p = torch.scalar_tensor(0.552)
