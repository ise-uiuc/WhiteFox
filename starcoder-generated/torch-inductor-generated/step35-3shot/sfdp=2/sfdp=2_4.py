
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout1(softmax_qk)
        output = dropout_qk.matmul(value)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 128)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)
