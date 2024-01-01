
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.attn_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1)) # Compute the dot product of the query and key, and scale it
        qk = qk + self.attn_mask # Add the attention mask to the scaled dot product
        attn_weight = torch.softmax(qk, dim=-1) # Apply softmax to the result
        output = attn_weight @ x2 # Compute the dot product of the attention weights and the value
        return output

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 2, 3, 4)
x2 = torch.randn(1, 4, 5, 6)
output = m(x1, x2)

