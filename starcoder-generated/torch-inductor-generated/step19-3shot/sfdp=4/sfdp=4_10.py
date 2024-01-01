
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, attn_mask):
        # Compute the scaled dot-product of the query and key tensors
        qk = q @ k.transpose(-2, -1)
        # Add the attention mask to the dot product
        qk = qk + attn_mask
        # Apply the softmax function to compute the attention weights
        attn_weight = torch.softmax(qk, dim=-1)
        # Compute the output
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 4, 256)
k = torch.randn(1, 4, 256)
v = torch.randn(1, 4, 256)
attn_mask = torch.zeros(1, 4, 4)
