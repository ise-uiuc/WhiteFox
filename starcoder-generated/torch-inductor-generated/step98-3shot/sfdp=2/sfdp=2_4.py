
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hidden_dim, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        # Compute the value dimension based on the input dimension, number of heads, and the hidden dimension
        self.value_dim = hidden_dim // num_heads
        self.w_q = torch.nn.Linear(input_dim, hidden_dim, bias=False) # Transform the input tensor from the input dimension to the hidden dimension
        self.w_k = torch.nn.Linear(input_dim, hidden_dim, bias=False) # Transform the input tensor from the input dimension to the hidden dimension
        self.w_v = torch.nn.Linear(input_dim, hidden_dim, bias=False) # Transform the input tensor from the input dimension to the hidden dimension
        self.w_o = torch.nn.Linear(hidden_dim, input_dim, bias=False) # Transform the input tensor from the hidden dimension to the input dimension
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2, x3=None):
        q = self.w_q(x1)
        k = self.w_k(x1)
        v = self.w_v(x1)
        if x2 is not None:
            k = self.w_k(x2)
            v = self.w_v(x2)
        if x3 is not None:
            k = self.w_k(x3)
            v = self.w_v(x3)
 
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.value_dim).permute(0, 2, 1, 3) # Reshape and permuted for batched matrix multiplication
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.value_dim).permute(0, 2, 1, 3) # Reshape and permuted for batched matrix multiplication
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.value_dim).permute(0, 2, 1, 3) # Reshape and permuted for batched matrix multiplication

        q *= self.value_dim ** -0.5 # Scale the value dimension
        # Compute the dot product of the value and the query, and then scale by the inverse square root of the value dimension
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk /= self.value_dim ** 0.5
 
        if self.training and 0.0 < self.dropout_p < 1.0:
            # Apply dropout
            qk = torch.nn.functional.dropout(qk, p=self.dropout_p)
 
        qk = self.softmax(qk) # Apply softmax to the scaled dot product
        output = torch.matmul(qk, v) # Compute the dot product of the softmax output and the value
 
        output = output.permute(0, 2, 1, 3) # Permuted for re-reshaping
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2] * output.shape[3]) # Reshape for dense output
        output = self.w_o(output) # Transform the input tensor from the hidden dimension to the input dimension
        return output

# Initializing the model
model = Model(input_dim=6, output_dim=2, num_heads=2, hidden_dim=4, dropout_p=0.8)

# Inputs to the model
x1 = torch.randn(4, 6)
x2 = torch.randn(4, 6)
x3 = torch.randn(4, 2)
