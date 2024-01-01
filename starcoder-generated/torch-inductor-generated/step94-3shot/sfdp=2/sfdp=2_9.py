
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.2):
        super().__init__()
        self.query = torch.nn.Linear(512, dim)
        self.key = torch.nn.Linear(512, dim)
        self.value = torch.nn.Linear(512, dim)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query_input, key_input, value_input, dropout_p=0.2):
        query = self.query(query_input)
        key = self.key(key_input)
        value = self.value(value_input)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(512 ** 0.25)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(512)

# Inputs to the model
query_input = torch.randn(8, 512)
key_input = torch.randn(8, 512)
value_input = torch.randn(8, 512)
print(__output__)
