
class Model(torch.nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
 
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
 
        self.query = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, input_tensor, input_mask):
        batch_size_tensor = input_tensor.size(0)
 
        query = self.query(input_tensor) # Apply the query linear transformation to the input tensor
        key = self.key(input_tensor) # Apply the key linear transformation to the input tensor
        value = self.value(input_tensor) # Apply the value linear transformation to the input tensor
 
        query = query.view(batch_size_tensor, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size_tensor, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size_tensor, -1, self.num_heads, self.head_dim)
 
        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size_tensor * self.num_heads, -1, self.head_dim)
        key = key.permute(0, 2, 1, 3).contiguous().view(batch_size_tensor * self.num_heads, -1, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous().view(batch_size_tensor * self.num_heads, -1, self.head_dim)
 
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of query and key
        scaled_qk = qk.div(self.scaling) # Scale the dot product by the scaling factor
        masking_matrix = input_mask.repeat(self.num_heads, 1, 1)  # Add a layer of repeated ones to the masking matrix as a bias
        dropped_scaled_qk = torch.nn.functional.dropout(scaled_qk, p=0.2, training=self.training) # Dropout the computed dot product
        softmax_qk = dropped_scaled_qk.softmax(dim=-1) # Apply softmax to the dot product
        masked_softmax_qk = softmax_qk.masked_fill(masking_matrix == 0, 0) # Mask the softmax output if the masking matrix has a zero
        dropout_qk = self.dropout(masked_softmax_qk) # Apply dropout to the masked softmax output
        return dropout_qk.matmul(value.view(batch_size_tensor * self.num_heads, -1, self.head_dim)).view(batch_size_tensor, -1, self.hidden_dim)

# Initializing the model
m = Model(2, input_dim)

# Inputs to the model
x1 = torch.randn(1, 4, input_dim)
h1 = torch.empty((1, 2, 4, input_dim // 2), requires_grad=True)
h2 = torch.empty((1, 2, 4, input_dim // 2), requires_grad=True)
h3 = torch.empty((1, 2, 4, input_dim // 2), requires_grad=True)
input_mask = torch.empty(1, 4, requires_grad=False)
input_mask[:, 3] = 1
