
class Model(torch.nn.Module):
    def __init__(self, num_heads, embedding_size, dropout_p):
        super().__init__()
        self.query_projection = torch.nn.Linear(embedding_size, embedding_size)
        self.key_projection = torch.nn.Linear(embedding_size, embedding_size)
        self.value_projection = torch.nn.Linear(embedding_size, embedding_size)
        self.num_heads = num_heads
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        q = q.reshape(q.shape[:-1] + (self.num_heads, q.shape[-1] // self.num_heads)).transpose(2, 3) 
        k = k.reshape(k.shape[:-1] + (self.num_heads, k.shape[-1] // self.num_heads)).transpose(2, 3)
        v = v.reshape(v.shape[:-1] + (self.num_heads, v.shape[-1] // self.num_heads)).transpose(2, 3)
        q_scaled = q.mul(scale_factor)
        softmax_qk = q_scaled.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        output = output.transpose(2, 3).reshape(output.shape[:-2] + (output.shape[2] * output.shape[3],))
        return output

# Initializing the model
dropout_p = 0.0
num_heads = 2
embedding_size = 24
m = Model(num_heads, embedding_size, dropout_p)

# Inputs to the model. The input tensors are initialized to uniform random distribution in the range [-5, 5].
query = torch.rand(10, embedding_size, embedding_size)
key = torch.rand(10, embedding_size, embedding_size)
value = torch.rand(10, embedding_size, embedding_size)
scale_factor = torch.rand(1)[0]
