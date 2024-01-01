
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(1, 1)
        self.key = torch.nn.Linear(1, 1)
        self.value = torch.nn.Linear(1, 2)
 
    def forward(self, x1, x2):
        q = self.query(x1) # Compute the query
        k = self.key(x2) # Compute the key
        v = self.value(x2) # Compute the value
        inv_scale_factor = np.sqrt(q.numel()) # Inverse scale factor is sqrt of the number of elements in the query
        qk = torch.matmul(q, k.transpose(-2, -1)) # Dot product of the query and key tensors
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product
        softmax_qk = torch.softmax(scaled_qk, dim=-1) # Apply softmax to the scaled dot product
        dropout_p = 0.5 # Dropout is applied with probability 0.5
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout
        o1 = dropout_qk.matmul(v) # Dot product of the dropout output and value
        return o1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 3, 16, 16)
x2 = torch.randn(128, 3, 16, 16)
