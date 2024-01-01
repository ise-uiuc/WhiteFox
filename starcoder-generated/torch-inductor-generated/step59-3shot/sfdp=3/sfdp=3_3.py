
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self,
                query,
                key,
                value,
                scale_factor=1 / np.sqrt(query.shape[-1])):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout_p=0.5)

# Inputs to the model
query = torch.randn(1, 5, 10)
key = torch.randn(1, 100, 5)
value = torch.randn(1, 100, 5)
