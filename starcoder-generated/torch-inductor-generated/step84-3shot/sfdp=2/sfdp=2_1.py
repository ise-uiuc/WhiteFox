
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.inv_scale_factor = 1 / np.sqrt(128)

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 128, 512)
x2 = torch.randn(64, 512, 128)
