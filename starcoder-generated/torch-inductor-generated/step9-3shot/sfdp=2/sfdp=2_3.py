
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout_qk = torch.nn.Dropout(p=0.1)
 
    def forward(self, qk, inv_scale_factor, dropout_p):
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
qk = torch.randn(128, 64)
inv_scale_factor = 0.5
dropout_p = 0.1
value = torch.randn(128, 128)
