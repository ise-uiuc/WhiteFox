
class Model(torch.nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_feedforward=2048, dropout_p=0.2):
        super().__init__()
        self.scale_factor = float(dim_model)^-0.5
        self.qkv = torch.nn.Linear(dim_model, dim_model * 3)
        self.p_dropout_fc = torch.nn.Linear(dim_model, dim_model)
        self.dropout = dropout_p
 
    def forward(self, query, key, value):
        qkv = self.qkv(query)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.p_dropout_fc(dropout(softmax_qk, p=self.dropout))
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 3, 512)
keys = torch.randn(1, 5, 512)
values = torch.randn(1, 5, 512)
