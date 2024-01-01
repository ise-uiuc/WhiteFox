
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = 1.0 / np.sqrt(query.shape[-1])
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.2
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 10, 10)
key = torch.randn(1, 10, 20, 20)
value = torch.randn(1, 10, 20, 20)
