
class Model(torch.nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p=0.1):
        q = q.view(q.size(0), q.size(1), self.n_hidden) # Reshape the query tensor
        k = k.view(k.size(0), k.size(1), self.n_hidden) # Reshape the key tensor
        v = v.view(v.size(0), v.size(1), self.n_hidden) # Reshape the value tensor
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
m = Model(256)

# Inputs to the model
q = torch.rand(5, 8, 256)
k = torch.rand(5, 16, 256)
v = torch.rand(5, 16, 256)
inv_scale_factor = torch.rand(n_head)
