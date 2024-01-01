
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=50, out_features=8, bias=True)
        self.linear2 = torch.nn.Linear(in_features=8, out_features=8, bias=True)
        self.linear3 = torch.nn.Linear(in_features=8, out_features=8, bias=True)
        self.linear4 = torch.nn.Linear(in_features=8, out_features=8, bias=True)
        self.gru1 = torch.nn.GRU(8, 16, 1)
 
    def forward(self, q, k, v, inv_scale_factor=1.0):
        q1 = self.linear1(q)
        k1 = self.linear2(k)
        v1 = self.linear3(v)
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v1)
        output = output.transpose(0, 1)
        _, hidden = self.gru1(output)
        return hidden[0]

# Initializing the model
m = Model()

# Inputs to the model
q, k, v, p, p = torch.randn(10, 8), torch.randn(16, 8), torch.randn(32, 8), torch.randn(50), torch.randn(10, 8)
