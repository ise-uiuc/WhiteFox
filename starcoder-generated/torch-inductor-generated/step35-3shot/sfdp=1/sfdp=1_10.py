
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.randn(1, 3, 10, 2) # (batch, n_head, sequence_length, hidden_size_per_head)
        self.key = torch.randn(1, 3, 20, 2) # (batch, n_head, sequence_length, hidden_size_per_head)
        self.value = torch.randn(1, 3, 20, 2) # (batch, n_head, sequence_length, hidden_size_per_head)
        self.inv_scale_factor = 30
        self.dropout_p = 0.1
 
    def forward(self, q, k, v):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 10, 2) # (batch, n_head, sequence_length, hidden_size_per_head)
key = torch.randn(1, 3, 20, 2) # (batch, n_head, sequence_length, hidden_size_per_head)
value = torch.randn(1, 3, 20, 2) # (batch, n_head, sequence_length, hidden_size_per_head)
