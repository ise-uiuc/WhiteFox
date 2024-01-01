
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input1, input2):
        QK = torch.matmul(query, key.transpose(-2, -1))
        v1 = QK.div(inv_scale_factor)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=dropout_p)
        v4 = v3.matmul(value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch_size, num_heads, sequence_length, embedding_size)
key = torch.randn(batch_size, num_heads, sequence_length, embedding_size)
value = torch.randn(batch_size, num_heads, sequence_length, embedding_size)
