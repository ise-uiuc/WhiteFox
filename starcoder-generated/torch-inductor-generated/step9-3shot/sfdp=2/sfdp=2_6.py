
class Model(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.query = torch.nn.Linear(512, num_heads * head_size)
        self.key = torch.nn.Linear(512, num_heads * head_size)
        self.value = torch.nn.Linear(512, num_heads * head_size)
 
    def forward(self, query, key, value, dropout_p=0.1, scale_factor=1.0):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(num_heads=4, head_size=64)

# Inputs to the model
query = torch.randn(1, 512)
key = torch.randn(2, 512)
value = torch.randn(2, 512)
