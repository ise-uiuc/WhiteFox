
class Model(torch.nn.Module):
    def __init__(self, input_tensor, hidden_size, num_heads, dropout_p, weight)
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
 
    def forward(self, q, k, v, mask=None):
        q = q.transpose(0, 1).transpose(1, 2).mul(0.05).round().div(0.05)
        k = k.transpose(0, 1).transpose(1, 2).mul(0.05).round().div(0.05)
        qkv = torch.matmul(q, self.weight) + torch.matmul(k, self.weight)
        attn = mask_logits(qkv)
        return attn

# Initializing the model
m = Model(hidden_size, 2)

# Inputs to the model
q = torch.randn(1, 10, hidden_size) # Set the query tensor as the input tensor
k = torch.randn(1, 20, hidden_size) # Set the key tensor as the input tensor
v = torch.randn(1, 20, hidden_size) # Set the value tensor as the input tensor
mask = None # Set the attention mask as the input tensor
