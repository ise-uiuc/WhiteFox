
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(Model, self).__init__()
        self.query = torch.nn.Linear(input_size, hidden_size)
        self.key = torch.nn.Linear(input_size, hidden_size)
        self.value = torch.nn.Linear(input_size, hidden_size)

    def forward(self, query, key, value, dropout_p=0.5, inv_scale_factor=1.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Inputs to the model
query = torch.randn(1, 2, 4, 4)
key = torch.randn(1, 2, 8, 8)
value = torch.randn(1, 2, 8, 8)
