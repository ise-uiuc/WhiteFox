
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=1000, embedding_dim=64)
        self.num_heads = 1
        self.head_size = 50
        # We are making sure that the size of the matrices embedding.weight and input (in the forward method) are multiples of num_heads and head_size respectively.
        assert self.embedding.embedding_dim % self.num_heads == 0, 'Embedding size should be a multiple of number of heads.'
        assert self.embedding.embedding_dim % self.head_size == 0, 'Head size should be a multiple of embedding size.'
        self.projection_size = self.head_size * self.num_heads
        self.query = torch.nn.Linear(self.projection_size, self.projection_size)
        self.key = torch.nn.Linear(self.projection_size, self.projection_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.value = torch.nn.Linear(self.projection_size, self.projection_size)
        self.output = torch.nn.Linear(self.projection_size, self.projection_size)
 
    def forward(self, x1):
        x2 = torch.nn.functional.embedding(x1, weight=self.embedding.weight) # Embedding
        x3 = torch.flatten(x2, start_dim=1) # Flatten the first two dimensions
        x4 = self.query(x3) # Apply linear transformation to the embedding tensor
        x5 = self.key(x3) # Apply linear transformation to the embedding tensor
        x6 = self.dropout(x3) # Apply dropout to the embedding tensor
        x7 = self.value(x3) # Apply linear transformation to the embedding tensor
        v1 = torch.matmul(x4, x5.transpose(-2, -1)) # Apply matrix multiplication to generate query-key dot product
        v2 = v1.mul(scale_factor) # Divide the dot product by a factor
        v3 = torch.nn.functional.softmax(v2, dim=-1) # Apply softmax to generate attention score
        v4 = self.dropout(v3) # Apply dropout to the softmax output
        v5 = torch.matmul(v4, x7) # Multiply the softmax output and value tensor
        v6 = v5.reshape(
            list(v5.size()[:-2]) +
            [self.num_heads, self.projection_size // self.num_heads]) # Reshape the value tensor
        v7 = torch.nn.functional.linear(v6, self.output.weight) # Apply linear transformation to perform attention projection
        out = v7.reshape(x2.size()) # Restore the dimension of the output
        out = self.output(out)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.LongTensor([[0, 1, 2, 3]]).cuda()
x2 = torch.randn((1, 4, 50)).cuda()
