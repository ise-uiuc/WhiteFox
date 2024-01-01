
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # None
    
    def inv_sqrt(self, x):
        return 1.0/torch.sqrt(x)
    
    def forward(self, query, key, scale_factor=1.0, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = self.inv_sqrt(scale_factor)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 6, 20)
key = torch.randn(5, 3, 20)
value = torch.randn(5, 3, 20)
