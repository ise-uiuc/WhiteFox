
class Model(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
 
        self.query_linear = torch.nn.Linear(3, num_heads * head_size)
        self.key_linear = torch.nn.Linear(2, num_heads * head_size)
        self.value_linear = torch.nn.Linear(2, num_heads * head_size)
 
        self.output_linear = torch.nn.Linear(num_heads * head_size, 2)
 
    def forward(self, x1):
        q = self.query_linear(x1) # Compute the query first in case of a multi-head attention
        k = self.key_linear(x2)
        v = self.value_linear(x2)
 
        q = q.view(q.size(0), self.num_heads, -1, self.head_size) # Reshape the query into multiple heads
        k = k.view(*k.size(), self.num_heads).permute(0, 1, 3, 2) # Reshape and permute the key
        v = v.view(*v.size(), self.num_heads).permute(0, 1, 3, 2)
 
        qk = torch.matmul(q, k) / math.sqrt(self.head_size) # Compute the scaled dot product of the query and key
 
        qk = qk + attn_mask.unsqueeze(0).unsqueeze(0) # Add the attention mask to the scaled dot product
        attn_weight = torch.softmax(qk, dim=-1) # Apply softmax to the attention weight
 
        output = torch.matmul(attn_weight, v) # Compute a weighted sum using the attention weights and the value
        output = output.transpose(1, 2).contiguous().reshape(x1.size(0), -1) # Reshape the output into a batch of 1D vectors
 
        output = self.output_linear(output)
        return output

# Initializing the model
num_heads = 4
head_size = 8
m = Model(num_heads, head_size)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 2, 128, 128)
