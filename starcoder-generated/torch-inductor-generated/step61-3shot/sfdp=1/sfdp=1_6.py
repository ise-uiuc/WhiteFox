
class Model(torch.nn.Module):
    def __init__(self, dropout_p, softmax_input_dim, softmax_dim):
        super().__init__()
        self.linear_qk = torch.nn.Linear(softmax_input_dim, softmax_input_dim)
        self.dropout_qk = torch.nn.Dropout(p=dropout_p)
        self.linear_output = torch.nn.Linear(softmax_input_dim, softmax_input_dim)

    def forward(self, query, key, value):
        qk = self.linear_qk(query) + self.linear_qk(key).transpose(-2, -1)
        qk = qk.div(math.sqrt(query.shape[-1]))
        dropout_qk = self.dropout_qk(torch.nn.functional.softmax(qk, dim=-1))
        output = self.linear_output(dropout_qk.matmul(value))
        return output

# Initializing the model
m = Model(0, softmax_input_dim, softmax_dim)

# Inputs to the model
x1 = torch.randn(1, 30, softmax_input_dim)
x2 = torch.randn(1, 40, softmax_input_dim)
x3 = torch.randn(1, 40, softmax_input_dim)
