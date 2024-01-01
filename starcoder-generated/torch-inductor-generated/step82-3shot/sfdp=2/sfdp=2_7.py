<fim_middle>
class Model(torch.nn.Module):
    def __init__(self):
        self.dropout_p = 0.5
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor=1.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model and their corresponding shapes
x1 = torch.randn(4, 32, 512)
x2 = torch.randn(4, 4, 512)
x3 = torch.randn(4, 4, 512)
