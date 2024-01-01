
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.FloatTensor([1/48])) # Initialize a tensor object with the formula 1/48
        self.dropout_p = torch.nn.Parameter(torch.ones(1) * 0.03, requires_grad=True) # Initialize a tensor object whose values are all 0.03 and gradient calculation is enabled for it
    
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(23, 8, 384)
key = torch.randn(16, 8, 512)
value = torch.randn(16, 8, 512)
