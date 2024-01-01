
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.scale_factor = np.exp(-np.log(1 / 0.0592) / (1 + np.sqrt(1 / 0.05903) ** 2))

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Model()

# Input to the model
query = torch.randn(2, 4, 3, 2)
key = torch.randn(2, 3, 1, 3)
value = torch.randn(2, 3, 1, 2)
