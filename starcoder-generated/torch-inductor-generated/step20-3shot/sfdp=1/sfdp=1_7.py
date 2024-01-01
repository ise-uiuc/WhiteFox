
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.query_size = input_size
        self.key_size = input_size

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.Linear(self.query_size, self.hidden_size)
        self.matmul2 = torch.nn.Linear(self.key_size, self.hidden_size)
        self.matmul3 = torch.nn.Linear(self.hidden_size, self.num_layers, bias=False)

    def forward(self, key, value, query, mask, inv_scale_factor):
        query = self.matmul1(query)
        key = self.matmul2(key)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.matmul3(torch.matmul(dropout_qk, value))
        return output

# Initializing the model
m = Model(21128, 768, 12, 0.1)

# Inputs to the model
query = torch.randn(768, 1)
key = torch.randn(768, 100)
value = torch.randn(768, 100)
mask = torch.empty((768, 100)).bernoulli_(1)
inv_scale_factor = torch.full((1,), 14000)
