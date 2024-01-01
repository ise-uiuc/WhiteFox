
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1, inv_scale_factor=1.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 8)
key = torch.randn(1, 4, 8)
value = torch.randn(1, 4, 11)
