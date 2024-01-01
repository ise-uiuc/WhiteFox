
class Model(torch.nn.Module):
    def __init__(self, N=64, D=2, H=2, dropout_p=0.2):
        super().__init__()
        self.qkv = torch.nn.Linear(D, D*H*3)
        self.softmax_kv = torch.nn.Softmax(dim=-1)
        self.dropout_kv = torch.nn.Dropout(dropout_p)
        self.final_linear = torch.nn.Linear(D*H, D)
 
    def forward(self, q, k, v, inv_scale_factor):
        qk = self.qkv(q)
        qk = qk.reshape(q.size(0), q.size(1), 3, int(qk.size(1)/3))
        q, k, v = qk.reshape(-1, int(qk.size(1)/3)), qk[:, :, 1, :], qk[:, :, 2, :]
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))/ inv_scale_factor
        softmax_qk = self.softmax_kv(scaled_qk)
        dropout_qk = self.dropout_kv(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        output = output.reshape(q.size(0), q.size(1), output.size(1))
        output = output.reshape(q.size(0), output.size(1), 1, 1)
        output = self.final_linear(output)
        return output
 
# Initializing the model
m = Model()
 
# Input tensors to the model
N, D, H, inv_scale_factor, dropout_p = 8, 256, 32, 0.01, 0.2
q = torch.randn(N, D)
k = torch.randn(N, D)
v = torch.randn(N, D)
