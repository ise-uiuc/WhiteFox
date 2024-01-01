
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        inv_scale_factor = math.sqrt((query.size(-1) * query.size(-2)))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        # output = dropout_qk.matmul(value)
        out = torch.matmul(dropout_qk, value)
        return out

# Initializing the model
q = np.random.random((1, 3, 640, 32))
k = np.random.random((1, 3, 32, 640))
v = np.random.random((1, 3, 640, 64))
query = torch.tensor(q).float().to(device)
key = torch.tensor(k).float().to(device)
value = torch.tensor(v).float().to(device)
m = Model()
