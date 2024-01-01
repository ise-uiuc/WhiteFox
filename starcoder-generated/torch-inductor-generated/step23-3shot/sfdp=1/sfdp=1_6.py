
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=0)
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q, k, v = torch.randn(10, 128, 512), torch.randn(10, 128, 512), torch.randn(10, 512, 1024)
inv_scale_factor = torch.randn(1)
dropout_p = 0.5
