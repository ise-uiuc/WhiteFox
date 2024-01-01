
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 8
        self.d_qk = 64
        self.d_v = 64
        self.d_model = self.n_head * self.d_qk
        self.dropout_p = 0.1
 
    def forward(self, input_tensor):
        x0 = self.dropout_p
        x1 = input_tensor
        x2 = x0 * self.d_model
        x3 = self.n_head * self.d_qk
        x4 = x2 / x3
        x5 = x1 * x4
        x6 = x5.view(16,12,2,16)
        x7 = torch.transpose(x6, 1, 2)
        x8 = torch.transpose(x7, 1, 3)
        x9 = torch.matmul(x8, x2)
        x10 = torch.nn.functional.dropout(x9, self.dropout_p, True)
        v11 = x2 * self.n_head
        v12 = x7.transpose(1, 2)
        v13 = x4 * self.d_v
        v14 = torch.matmul(v12, v13)
        v15 = torch.transpose(v14, 1, 2)
        v16 = v11 / v15.transpose(1, 2).size()[1]
        v17 = v10 * v16
        return v17

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(16,12,128)
