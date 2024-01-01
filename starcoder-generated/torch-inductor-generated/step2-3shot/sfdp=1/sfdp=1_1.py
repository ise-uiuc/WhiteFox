
class Model(torch.nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
 
        self.in_proj_weight = torch.nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        torch.nn.init.xavier_uniform_(self.in_proj_weight)
 
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = float(np.sqrt(float(self.embed_dim)))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output
 