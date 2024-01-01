
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.query_projection = torch.nn.Linear(64, dim)
        self.key_projection = torch.nn.Linear(64, dim)
        self.value_projection = torch.nn.Linear(64, dim)
        self.positional_embedding = torch.nn.Embedding(32, dim)
        self.dropout_p = dropout_p
 
 
    def forward(self, x1, x2):
        dim = self.positional_embedding.embedding_dim # Get the dimension of the positional encoding
        t = x2.shape[1] # Get the temporal length of the input tensor from the shape of the input tensor
        t = t // dim # Calculate the number of timesteps by dividing the temporal length of the input tensor by the dimension of the positional encoding
        t = self.make_positions(t) # Create timesteps using the function created in the above section
        t = t.to(x2.device) # Put the timesteps on the same device as the input tensor
        positional_encoding = self.positional_embedding(t) # Encode timesteps to be added into the input tensor
        x2 = x2 + positional_encoding # Add the timesteps into the input tensor
        query = self.query_projection(x1) # Transform input tensor by a linear layer for query
        key = self.key_projection(x2) # Transform input tensor by a linear layer for key
        value = self.value_projection(x2) # Transform input tensor by a linear layer for value
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of one of the input matrices
        inv_scale_factor = torch.rsqrt(torch.tensor(key.shape[0]).float()) # Calculate the inverse to the square root of the dimension of the key
        scaled_qk = qk * 0.5 * inv_scale_factor # Scale the dot product by a constant
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax on the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product between the dropout output and the value
        return output
 
    def make_positions(self, seq_len):
        output = []
        for p in range(seq_len):
            for j in range(self.positional_embedding.embedding_dim):
                output.append(p * self.positional_embedding.embedding_dim + j)
        return torch.tensor(output).unsqueeze(1) # Convert the output into a column vector

# Initializing the model
m = Model(dim=64, num_heads=4, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
