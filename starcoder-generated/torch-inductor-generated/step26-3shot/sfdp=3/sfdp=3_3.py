
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor=1.0, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 50, 150)
key = torch.randn(1, 1, 50, 250)
value = torch.randn(1, 1, 50, 250)
