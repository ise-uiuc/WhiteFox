
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.softmax_dim = 1
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=self.softmax_dim)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
inv_scale_factor = torch.randn(query.shape[0], 1, 1)
x1 = torch.randn(128, 64, 1024)
x2 = torch.randn(128, 256, 256)
x3 = torch.randn(128, 64, 256)
