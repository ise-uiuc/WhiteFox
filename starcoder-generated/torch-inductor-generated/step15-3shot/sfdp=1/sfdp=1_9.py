
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout_qk = torch.nn.Dropout(p=0.1)
 
    def forward(self, q, k, v, inv_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = self.softmax_qk(scaled_qk) # Apply softmax to the scaled dot product
        dropout_qk = self.dropout_qk(softmax_qk) # Apply dropout to the softmax output
        return dropout_qk.matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 100, 2304)
k = torch.randn(1, 100, 4608)
v = torch.randn(1, 100, 4608)
inv_scale_factor = torch.randn((1, 1, 1, 768))
