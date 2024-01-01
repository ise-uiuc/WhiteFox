
class Model(torch.nn.Module):
    def __init__(self, dropout_p, query_emb_dim, content_emb_dim, out_emb_dim):
        super().__init__()
        self.dropout_p = dropout_p
        self.query = torch.nn.Parameter(torch.empty((query_emb_dim,)))
        self.key = torch.nn.Parameter(torch.empty((content_emb_dim, query_emb_dim)))
        self.value = torch.nn.Parameter(torch.empty((content_emb_dim,)))
        self.fc = torch.nn.Linear(self.value.shape[0], out_emb_dim, bias=False)
        self.scale_factor = self.key.shape[0]**(-0.5)
 
    def forward(self, q, contents):
        qk = torch.matmul(q, self.key.transpose(-2, -1))
        v = self.fc(self.value)
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
 
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model(dropout_p=0.135, query_emb_dim=16, content_emb_dim=64, out_emb_dim=512)

# Inputs to the model
q = torch.randn(8, 16)
contents = torch.randn(117, 8, 64)
