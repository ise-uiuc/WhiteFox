
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4096, 512)
        self.q_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.k_weight = torch.nn.Parameter(torch.randn(512, 512))
        self.v_weight = torch.nn.Parameter(torch.randn(512, 512))
    
    def forward(self, inputs, attn_mask, dropout_p):
        self.qkv = self.fc1(inputs)
        self.qkv = self.qkv.reshape(-1, 4096, 512)
        self.q = torch.matmul(self.qkv, self.q_weight)
        self.k = torch.matmul(self.qkv, self.k_weight)
        self.v = torch.matmul(self.qkv, self.v_weight)
        self.q = self.q / math.sqrt(4096)
        self.weight = self.q + attn_mask
        self.weight = torch.softmax(self.weight.reshape(-1, 512), dim=-1)
        self.dropout = self.weight.reshape(-1, 512)
        self.dropout = torch.dropout(self.weight, dropout_p)
        return torch.matmul(self.dropout, self.v)
    
# Initializing the model
m = Model()
# Inputs to the model
inputs = torch.randn(1, 4096)
attn_mask = torch.randn(512)
dropout_p = 0.5
