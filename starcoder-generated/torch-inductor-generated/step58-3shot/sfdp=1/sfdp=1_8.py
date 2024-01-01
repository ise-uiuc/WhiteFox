
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(12, 8)
        self.fc2 = torch.nn.Linear(8, 6)

    def forward(self, q1, k2, v2, scale_factor, dropout_p):
        q2 = self.fc1(q1)
        k3 = self.fc1(k2)
        v3 = self.fc2(v2)
        scaled_qk = torch.matmul(q2, k3.transpose(-2, -1))
        scaled_qk = scaled_qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(20, 12)
k2 = torch.randn(20, 12)
v2 = torch.randn(20, 6)
scale_factor = torch.randn(8, 8)
dropout_p =.3
