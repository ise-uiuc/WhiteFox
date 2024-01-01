
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(1)
 
    def forward(self, query, key, value, scale_factor=1):
        qk = torch.matmul(query, key.transpose(-2,-1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output 

# Initializing the model
m = Model()

# Generated input tensors
query = torch.randn(1, 2, 3) 
key = torch.randn(1, 3, 4) 
value = torch.randn(1, 3, 4)
