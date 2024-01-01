
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p, inv_scale_factor):
        super().__init__()
        self.query_dim = query_size
        self.key_dim = key_size
        self.value_dim = value_size
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
    
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax ouput
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model(query_size=512, key_size=512, value_size=512, dropout_p=0.1, inv_scale_factor=math.sqrt(0.2))

# Inputs to the model
query = torch.randn(1, 512, 16)
key = torch.randn(1, 512, 320)
value = torch.randn(1, 512, 320)
