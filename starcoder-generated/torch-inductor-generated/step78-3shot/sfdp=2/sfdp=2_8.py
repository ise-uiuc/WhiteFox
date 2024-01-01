
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads=2, hidden_size=4):
        super().__init__()
        self.query = torch.nn.Linear(num_attention_heads, hidden_size, bias=False)
        self.key = torch.nn.Linear(num_attention_heads, hidden_size, bias=False)
        self.value = torch.nn.Linear(num_attention_heads, hidden_size, bias=False)
 
    def forward(self, query, key, value, dropout_p=0.8, inv_scale_factor=1.0):
        v1 = self.query(query)
        v2 = self.key(key)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.div(inv_scale_factor)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        output = torch.nn.functional.dropout(v5, p=dropout_p)
        output = self.value(output).matmul(value)
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(1, 1, 4)
key = torch.randn(1, 2, 4)
value = torch.randn(1, 2, 4)
