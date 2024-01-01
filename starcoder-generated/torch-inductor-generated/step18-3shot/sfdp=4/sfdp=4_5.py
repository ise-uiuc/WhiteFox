
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Project queries, keys and values
        # The linear projections of queries, keys and values are shared
        self.project = nn.Linear(hidden_size, hidden_size)

        # Learned diagonal attention mask
        self.mask = Mask diagonal

    def forward(self, query, key, value):
        # Apply the linear projections to the queries, keys and values
        query, key, value = self.project(query), self.project(key), self.project(value)

        # Compute the dot products between the queries and keys
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot products by a factor of sqrt(query.size(-1))
        scores /= math.sqrt(query.size(-1))
        
        # Apply the attention mask
        scores += self.mask
        
        # TODO: Apply a softmax function to the scaled dot products
        attn = None
        
        # Compute the weighted sum of the values by using
        # the attention weights as weights and the values
        # as a Tensor
        output = None
        
        return output
# Initializing the model
m = Attention(hidden_size=64)
m.apply(init_weights)

# Inputs to the model
query = torch.randn(32, 50, 64)
key = torch.randn(32, 64, 64)
value = torch.randn(32, 64, 50)
