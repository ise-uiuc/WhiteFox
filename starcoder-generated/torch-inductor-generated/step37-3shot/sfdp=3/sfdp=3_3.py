
class Model(torch.nn.Module):
    def __init__(self, d_model=8, nheads=4, dropout_p=0.8):
        super().__init__()
        self.query_affine1 = torch.nn.Linear(d_model, d_model, bias=True)
        self.key_affine1 = torch.nn.Linear(d_model, d_model, bias=True)
        self.value_affine1 = torch.nn.Linear(d_model, d_model, bias=True)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
 
        self.dropout1 = torch.nn.Dropout(dropout_p)
 
        self.query_affine2 = torch.nn.Linear(d_model, nheads, bias=False)
        self.key_affine2 = torch.nn.Linear(d_model, nheads, bias=False)
        self.value_affine2 = torch.nn.Linear(d_model, nheads, bias=False)
 
        self.query_affine3 = torch.nn.Linear(nheads, d_model, bias=False)
        self.key_affine3 = torch.nn.Linear(nheads, d_model, bias=False)
        self.value_affine3 = torch.nn.Linear(nheads, d_model, bias=False)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
 
    def forward(self, x1):
        v1 = self.query_affine1(x1)
        v2 = self.key_affine1(x1)
        v3 = self.value_affine1(x1)
 
        t1 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = t1 / np.sqrt(v1.size(-1))
        t2 = self.dropout1(v4)
 
        v5 = self.query_affine2(t2)
        v6 = self.key_affine2(t2)
        v7 = self.value_affine2(t2)
        v8 = v5.transpose(-2, -1)
        v9 = torch.matmul(v8, v7)
 
        t3 = v9 / np.sqrt(v5.size(-1))
        t4 = self.query_affine3(t3)
        t5 = self.key_affine3(t3)
        t6 = self.value_affine3(t3)
        v10 = t4 * v1
        v11 = t5 * v3
        v12 = t6 * v2
 
        v13 = v10 + v11
        v14 = v13 + v12
        v15 = self.layer_norm1(v14)
 
        v16 = self.query_affine3(v15) + v10
        v17 = self.key_affine3(v15) + v11
        v18 = self.value_affine3(v15) + v12
        v19 = self.layer_norm2(v16 + v17 + v18)
 
        return v19

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
