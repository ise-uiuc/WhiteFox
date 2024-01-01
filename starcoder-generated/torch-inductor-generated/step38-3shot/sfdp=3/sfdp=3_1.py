
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.scale_factor = np.sqrt(1.0 / 512.0)
        self.qk_proj = torch.nn.Linear(512, 512)
        self.value_proj = torch.nn.Linear(512, 512)
        self.tanh = torch.nn.Tanh()
        self.softmax_qk = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        qk = self.tanh(self.qk_proj(x1))
        qk = self.tanh(self.value_proj(x2))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(1, 512)
