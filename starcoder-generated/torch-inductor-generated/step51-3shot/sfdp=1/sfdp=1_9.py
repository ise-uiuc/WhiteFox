
class Model(torch.nn.Module):
    def __init__(self, dim_model=32, dim_key=8, dim_value=2):
        super().__init__()
        self.fc11 = torch.nn.Linear(dim_model, dim_key, bias=False)
        self.fc21 = torch.nn.Linear(dim_model, dim_key, bias=False)
        self.fc12 = torch.nn.Linear(dim_model, dim_value, bias=False)
        self.fc22 = torch.nn.Linear(dim_model, dim_value, bias=False)
 
    def forward(self, x1, x2, q1, q2, dropout_p=0.25):
        qh1 = self.fc11(q1)
        qh2 = self.fc21(q2)
        kh1 = self.fc12(x1)
        kh2 = self.fc22(x2)
        qk = torch.matmul(qh1, kh1.transpose(-2, -1)) + torch.matmul(qh2, kh2.transpose(-2, -1))
        inv_scale_factor = np.power(qk.size(-1), -0.5)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(kh1) + dropout_qk.matmul(kh2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
q1 = torch.randn(1, 8, 16, 16)
q2 = torch.randn(1, 8, 16, 16)
