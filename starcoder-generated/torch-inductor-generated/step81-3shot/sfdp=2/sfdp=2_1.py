
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.8
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(1, 2))
        inv_scale_factor = np.float32(1.0 / np.sqrt(x1.shape[-1]))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) - 1
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x3)
        return output

# Inputs to the model
x1 = torch.randn(1, 4, 14)
x2 = torch.randn(1, 14, 16)
x3 = torch.randn(1, 4, 16)
