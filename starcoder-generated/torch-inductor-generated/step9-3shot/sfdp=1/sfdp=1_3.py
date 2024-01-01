
class Model(torch.nn.Module):
    def __init__(self, scale_factor=math.sqrt(1.0), dropout_p=0.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
        self.dense = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(p=dropout_p, inplace=False)
        self.dense2 = torch.nn.Linear(1024, 1024)
 
    def forward(self, x1, x2, x3):
        v1 = self.dense(x1)
        v2 = self.dropout(v1)
        v3 = self.dense2(v2)
        k1 = self.dense(x2)
        k2 = self.dropout(k1)
        k3 = self.dense2(k2)
        v = v3
        k = k3
        s = self.scale_factor
        d = self.dropout_p
        q = v3 
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(s) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=d) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
x2 = torch.randn(1, 1024)
x3 = torch.randn(1, 1024)
