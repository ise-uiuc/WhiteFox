
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, input_tensor):
        q = torch.randn(3, 2)
        k = torch.randn(2, 4)
        v = torch.randn(2, 4)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(q.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(3, 2)
