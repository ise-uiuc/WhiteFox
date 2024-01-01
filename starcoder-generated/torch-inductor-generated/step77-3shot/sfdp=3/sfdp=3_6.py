
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.Tensor([1536.0])
        
    def forward(self, query, key, value, mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1024, 1024)
key = torch.randn(1, 1024, 1024)
value = torch.randn(1, 1024, 1024)
