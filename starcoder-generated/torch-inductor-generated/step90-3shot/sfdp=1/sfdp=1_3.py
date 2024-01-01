
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # You are free to change the size
        self.q = torch.nn.Parameter(torch.randn(5, 10, 10), requires_grad=True)
        self.k = torch.nn.Parameter(torch.randn(5, 10, 10), requires_grad=True)
        self.v = torch.nn.Parameter(torch.randn(5, 10, 20), requires_grad=True)
        # You are free to change the value of the scale factor
        self.inv_scale_factor = 1.0
 
    def forward(self, query, key):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(self.inv_scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = F.dropout(softmax_qk, p=0.5)
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(self.v)
        return output    

# Initializing the model
m = Model()
# Setting up input values
query = torch.randn(5, 3, 10)
key = torch.randn(5, 3, 10)
