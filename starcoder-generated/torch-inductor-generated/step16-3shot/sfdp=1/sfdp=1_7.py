
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, inputs):
        q = torch.randn(len(inputs), 8, 8)
        k = torch.randn(len(inputs), 8, 8)
        v = torch.randn(len(inputs), 8, 8)
        scale_factor = 4
        dropout_p = 0.7
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(scale_factor) # Scale the dot product by an inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
inputs = [torch.randn(1, 8, 8) for _ in range(10)]
