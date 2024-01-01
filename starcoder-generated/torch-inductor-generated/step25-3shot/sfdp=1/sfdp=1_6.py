
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1, inv_scale_factor=1/8):
        super().__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
        self.k = [[1, 2], [3, 4]]
        self.q = [[3, 4], [1, 2]]
        self.v = [[10, 30], [20, 40]]       
 
    def forward(self, x):
        qk = torch.matmul(self.q, self.k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, self.v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 2)
