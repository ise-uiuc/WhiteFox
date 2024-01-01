
class Model(torch.nn.Module):
    def __init__(self, heads_num, d_model, kernel_size, dropout_p):
        super().__init__()
        self.Wq = torch.nn.Conv2d(heads_num, d_model // heads_num, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.Wk = torch.nn.Conv2d(heads_num, d_model // heads_num, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.Wv = torch.nn.Conv2d(heads_num, d_model // heads_num, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v):
        Q = torch.relu(self.Wq(q)) # Apply ReLU to the output of the query convolution
        K = torch.relu(self.Wk(k)) # Apply ReLU to the output of the key convolution
        V = torch.relu(self.Wv(v)) # Apply ReLU to the output of the value convolution
        output = self.dropout(Q@K.transpose(-2, -1) / np.sqrt(K.size(-1))) # Compute the scaled dot product
        output = output@V # Compute the dot product of the scaled dot product and the value
        return output
 
# Initializing the model
m = Model(heads_num=2, d_model=8, kernel_size=3, dropout_p=0.4567)
# Inputs to the model
q = torch.randn(1, 2, 64, 64)
k = torch.randn(1, 2, 64, 64)
v = torch.randn(1, 2, 64, 64)
