
class Model(torch.nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.query = torch.nn.Linear(input_size, input_size)
        self.key = torch.nn.Linear(input_size, input_size)
        self.value = torch.nn.Linear(input_size, input_size)

    def forward(self, query, key, value, scale_factor, dropout_p):
        q = self.query(query)
        k = self.key(key)
        v = value
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(128)

# Input to the model
query = torch.randn(8, 128, 2)
key = torch.randn(8, 256, 2)
value = torch.randn(8, 256, 2)
scale_factor = query.shape[-1]**(-0.5)
dropout_p = 0.2
