

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_sf = float(64)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk
        scaled_qk = scaled_qk / self.inv_sf
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 1)
x2 = torch.randn(1, 8, 1)
