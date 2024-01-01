
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = np.random.randint(0, 100, (10, 32))
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(32) # This value should be derived from the weight of the model, and is usually constant
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.from_numpy(np.random.randint(0, 100, (10, 32))).float()
k = torch.from_numpy(np.random.randint(0, 100, (15, 32))).float()
v = torch.from_numpy(np.random.randint(0, 100, (15, 32))).float()
