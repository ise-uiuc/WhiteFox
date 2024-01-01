
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key = torch.nn.Linear(hidden_size, hidden_size)
        self.query_value = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x):
        kv1 = self.query_key(x)
        v1 = self.query_value(x)
        qk = torch.matmul(kv1, v1.transpose(-2, -1))
        scale_factor = query.size(-1) ** -0.5
        scaled_qk = qk.div(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        result = dropout_qk.matmul(v1)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, len_seq, hidden_size)
