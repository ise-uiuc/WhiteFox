
class Model(torch.nn.Module):
    def forward(self, query, key, value, dim=-1, scale_factor=1.0, dropout_p=0.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=dim)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.rand(1, 3, 4, 32)
key = torch.rand(1, 3, 32, 16)
value = torch.rand(1, 3, 4, 16)
dim = -1
scale_factor = 10
dropout_p = 0.8
