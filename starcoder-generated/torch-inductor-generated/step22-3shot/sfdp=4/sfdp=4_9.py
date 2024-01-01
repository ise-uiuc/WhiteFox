
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attention_mask):
        qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        qk = qk + attention_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, value)
        return output

# Initialize the model. The size of the query, key, and value are 3*5, 4*5 and 5*7, respectively.
m = Model()

# Generate random input tensors of size 3*5, 4*5 and 5*7.
query = torch.randn(3, 5)
key = torch.randn(4, 5)
value = torch.randn(5, 7)
attention_mask = torch.zeros((3, 4))

# Inputs to the model
