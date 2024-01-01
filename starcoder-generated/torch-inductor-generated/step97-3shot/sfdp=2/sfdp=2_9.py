
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        # Use H=1 and dmodel=1 to simplify the problem size
        self.query = torch.nn.Linear(1, 1, bias=False)
        self.key = torch.nn.Linear(1, 1)
        self.value = torch.nn.Linear(1, 1)
 
    def forward(self, query, key, value, dropout_p=0.2):
        inv_scale_factor = 2.0 / (key.shape[-2] ** 2)
        q, k, v = self.query(query), self.key(key), self.value(value)
        # Apply self attention
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
query   = torch.randn(1, 1, 1)
key     = torch.randn(1, 1, 1)
value   = torch.randn(1, 1, 1)
