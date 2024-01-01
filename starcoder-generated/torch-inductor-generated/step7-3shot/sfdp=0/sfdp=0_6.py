
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, dim)
        self.w2 = torch.nn.Linear(dim, dim)
 
    def forward(self, q, k, v, inv_scale=None):
        if inv_scale is not None:
            # Set the inverse square root of `inv_scale` as the parameter `divider` for the Softmax operator.
            q_key = q.matmul(k.transpose(-2, -1)) / inv_scale
        else:
            q_key = q.matmul(k.transpose(-2, -1))
        attention_probs = torch.nn.Softmax(dim=-1)(q_key)
        output = attention_probs.matmul(v)
        return output
 
# Initializing the model
m = Model(256)

# Inputs to the model
q = torch.randn(1, 8, 256)
k = torch.randn(1, 16, 256)
v = torch.randn(1, 16, 256)

