
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, mask=None):
        q = q.unsqueeze(1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 2.7 - 0.7 * random.random() # Randomly get an inverse scale factor
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1).masked_fill(mask==0, -np.inf) # Apply softmax to the scaled dot product, and mask invalid elements
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v.type_as(dropout_qk))
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 32, 64)
k = torch.randn(2, 32, 64)
v = torch.randn(2, 32, 64)
mask = torch.zeros(2, 32)
