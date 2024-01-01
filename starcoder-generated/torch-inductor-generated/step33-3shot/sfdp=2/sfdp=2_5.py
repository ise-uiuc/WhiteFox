
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_p):
        super().__init__()
        self.embedding_dim = embedding_dim # Embedding's dimension
        self.num_heads = num_heads # Number of heads
        self.dropout_p = dropout_p # Probability of dropout
        self.W_q = torch.nn.Linear(embedding_dim, num_heads * embedding_dim, bias=False)
        self.W_k = torch.nn.Linear(embedding_dim, num_heads * embedding_dim, bias=False)
        self.W_v = torch.nn.Linear(embedding_dim, num_heads * embedding_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, key_padding_mask, training):
        batch_size = query.shape[0]
        q = self.W_q(query) # Generate queries
        k = self.W_k(key) # Generate keys
        v = self.W_v(value) # Generate values
        q = q.reshape(batch_size, q.shape[1], self.num_heads, -1) # Reshape the queries into batches and heads
        k = k.reshape(batch_size, k.shape[1], self.num_heads, -1) # Reshape the keys into batches and heads
        v = v.reshape(batch_size, v.shape[1], self.num_heads, -1) # Reshape the values into batches and heads
        scaled_dot_product = torch.einsum('bhie,bhje->bhij', q, k) # Compute the dot product of the queries and the keys
        inv_scale_factor = 1 / math.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))
        scaled_dot_product = scaled_dot_product.div(inv_scale_factor) # Scale the dot products by the inverse scale factor
        softmax_scaled_dot_product = scaled_dot_product.softmax(dim=-1) # Apply softmax to the scaled dot products
        dropout_softmax_scaled_dot_product = self.dropout(softmax_scaled_dot_product) # Apply dropout to the softmax scaled dot products
        output = torch.matmul(dropout_softmax_scaled_dot_product, v) # Compute the dot product of the dropout softmax scaled dot products and the values
        return output.reshape(batch_size, output.shape[1], -1) # Reshape the computed dot products into batches and queries

# Initializing the model
embedding_dim = 64 # Set the embedding dimension
num_heads = 2 # Set the number of heads
dropout_p = 0.1 # Set the probability of dropout
m = Model(embedding_dim, num_heads, dropout_p)

# Inputs to the model
query = torch.randn(8, 3, 64)
key = torch.randn(8, 10, 64)
value = torch.randn(8, 10, 64)
key_padding_mask = torch.tensor([ [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]] ])
training = True
