
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.8
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.dropout.p = 0.8

        self.inv_scale_factor = np.sqrt(1.0/(1024*1024))

    def forward(self, x3, x4):
        v8 = torch.matmul(x3, x4.transpose(-2, -1))
        v9 = v8.div(self.inv_scale_factor)
        v10 = torch.nn.functional.softmax(v9, dim=-1)
        v11 = self.dropout(v10)
        v12 = v11.matmul(x4)
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 1024, 36)
x4 = torch.randn(1, 64, 1024)
