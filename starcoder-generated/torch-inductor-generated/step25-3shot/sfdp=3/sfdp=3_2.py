
class Model(torch.nn.Module):
    def __init__(self, num_heads, query_size, key_size, value_size, input_size):
        super().__init__()
        self.query = torch.nn.Linear(query_size, num_heads * key_size)
        self.key = torch.nn.Linear(key_size, num_heads * key_size)
        self.value = torch.nn.Linear(value_size, num_heads * value_size)
        self.output = torch.nn.Linear(num_heads * value_size, input_size)
        self.dropout_p = 0.75
        self.scale_factor = 1 / (query_size ** 0.5)
 
    def forward(self, x1, x2, x3):
        query = self.query(x1)
        key = self.key(x2)
        value = self.value(x3)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = self.output(dropout_qk.matmul(value))
        return output

# Initializing the model
m = Model(num_heads=4, query_size=32, key_size=32, value_size=32, input_size=32)

# Inputs to the model
x1 = torch.randn(1, 8, 32)
x2 = torch.randn(1, 8, 32)
x3 = torch.randn(1, 8, 32)
