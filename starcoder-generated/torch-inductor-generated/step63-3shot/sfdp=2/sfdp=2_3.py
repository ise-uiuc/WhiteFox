
class Model(torch.nn.Module):
    def __init__(self, nhead, batch_size, num_patches, dim_model, dim_feedforward, dropout_p):
        super().__init__()
        self.nhead = nhead
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout_p
 
        self.norm1 = torch.nn.LayerNorm(self.dim_model)
        self.q = torch.nn.Linear(self.dim_model, self.dim_model)
        self.k = torch.nn.Linear(self.dim_model, self.dim_model)
        self.v = torch.nn.Linear(self.dim_model, self.dim_model)
        self.scale = self.dim_model ** -0.5
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.dim_model, self.dim_feedforward), \
            torch.nn.ReLU(), \
            torch.nn.Linear(self.dim_feedforward, self.dim_model), \
            torch.nn.Dropout(self.dropout_p))
 
    def forward(self, x1):
        x1 = self.norm1(x1)
        q = self.q(x1)
        k = self.k(x1)
        v = self.v(x1)
        batch_size = x1.shape[0]
        heads_num = len(self.nhead)
        dim_model = self.dim_model
        num_patches = self.num_patches
 
        q = q.view(batch_size, heads_num, num_patches, dim_model).permute(0, 2, 1, 3)
        k = k.view(batch_size, heads_num, num_patches, dim_model).permute(0, 2, 3, 1)
        v = v.view(batch_size, heads_num, num_patches, dim_model).permute(0, 2, 1, 3)
        scaled_qk = torch.matmul(q, k) * self.scale
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        dropout_qk = dropout_qk.permute(0, 2, 1, 3)
        output = torch.matmul(dropout_qk, v)
        nfeatures = output.shape[2]
        output = output.view(batch_size, nfeatures, heads_num * dim_model)
        output = self.mlp(output)
        return output

# Initializing the model
m = Model(nhead=[1, 2], batch_size=2, num_patches=4, dim_model=16, dim_feedforward=64, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(2, 4, 16)
