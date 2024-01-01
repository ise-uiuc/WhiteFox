
class Model(torch.nn.Module):
    def __init__(self, dropout_p, inv_scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the input inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model(0.125, 0.10000000149011612)

# Inputs to the model
query = torch.randn(4, 64, 64)
key = torch.randn(4, 64, 64)
value = torch.randn(4, 64, 64)
