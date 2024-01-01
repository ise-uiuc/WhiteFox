
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, drop_p=0.1):
        super().__init__()
        self.query = torch.nn.Linear(dim, dim, bias=True)
        self.key = torch.nn.Linear(dim, dim, bias=True)
        self.value = torch.nn.Linear(dim, dim, bias=True)
        self.scale_factor = (dim / num_heads)**-0.5
        self.dropout_p = drop_p
 
    def forward(self, query, key, value):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initialize the model
dim = 512
num_heads = 2
m = Model(dim, num_heads, drop_p=0.1)

# Inputs to the model
query = torch.randn(1, dim)
key = torch.randn(1, dim)
value = torch.randn(1, dim)
