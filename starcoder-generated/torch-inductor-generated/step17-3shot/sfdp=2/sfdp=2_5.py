
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(8, 8)
        self.key = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2, x3, x4):
        q1 = self.query(x1)
        k1 = self.key(x2)
        v1 = self.value(x3)
        inv_scale_factor = x4
        # Compute the dot product of the query and the key
        qk1 = torch.matmul(q1, k1.transpose=-2, -1)
        # Scale the dot product by the inverse scale factor
        scaled_qk1 = qk1.div(inv_scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk1 = scaled_qk1.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk1 = torch.nn.functional.dropout(softmax_qk1, p=0.5)
        # Compute the dot product of the dropout output and the value
        output1 = dropout_qk1.matmul(v1)
        return output1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
x3 = torch.randn(1, 8)
x4 = torch.tensor([0.5])
