
class Model(torch.nn.Module):
    def __init__(self, dropout_p, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_weight = torch.nn.Parameter(torch.randn(num_heads, dim, dim)) # q_weight parameter of shape num_heads x dim x dim
        self.k_weight = torch.nn.Parameter(torch.randn(num_heads, dim, dim)) # k_weight parameter of shape num_heads x dim x dim
        self.v_weight = torch.nn.Parameter(torch.randn(num_heads, dim, dim)) # v_weight parameter of shape num_heads x dim x dim
        self.out_weight = torch.nn.Parameter(torch.randn(num_heads, dim, dim)) # out_weight parameter of shape num_heads x dim x dim
        self.bias = torch.nn.Parameter(torch.randn(num_heads, dim)) # bias parameter of shape num_heads x dim
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1):
        q, k, v = torch.chunk(x1, self.num_heads, dim=1) # Split the input into the query, key and value tensors
        q = torch.matmul(q, self.q_weight.transpose(-2, -1)) # Compute the dot product of the query and the q_weight tensor
        k = torch.matmul(k, self.k_weight.transpose(-2, -1)) # Compute the dot product of the key and the k_weight tensor
        v = torch.matmul(v, self.v_weight.transpose(-2, -1)) # Compute the dot product of the value and the v_weight tensor
        scaled_qk = q.softmax(dim=2) * k.softmax(dim=2) # Element-wise multiplication of the output of the softmax on query and softmax on key
        scaled_qk = scaled_qk.softmax(dim=2) * k.softmax(dim=2) # Element-wise multiplication of the scaled_qk tensor and the softmax on key
        scaled_qk = scaled_qk.softmax(dim=2) # softmax on the scaled_qk tensor
        out = self.dropout(scaled_qk).matmul(self.value) # Compute the dot product of the dropout output and the out_weight tensor
        out = out.transpose(1, 2) # Transpose the output tensor on the first and second dimension
        out = out.reshape(-1, self.dim * self.num_heads) # Concatenate the first dimension with the second dimension
        out = torch.matmul(out, self.out_weight.transpose(0, 1)) + self.bias # Matrix multiple the out_weight tensor and the concatenate tensor and then add the bias tensor
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 64, dim)
