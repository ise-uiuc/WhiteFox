
class Conv(torch.nn.Module):
    def __init__(self, qk_depth, v_depth):
        super().__init__()
        self.depth = max(qk_depth, v_depth) // 2
        self.qk_depth = qk_depth
        self.v_depth = v_depth

        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(self.qk_depth, self.depth, 1, bias=False),
            torch.nn.BatchNorm2d(self.depth),
            torch.nn.ReLU(inplace=True))
 
        self.conv = torch.nn.Conv2d(self.depth, self.depth, 1, bias=False)
 
        self.reduce = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.depth),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(self.depth, self.v_depth, 1))

        self.unproject_v = torch.nn.Conv2d(self.depth, self.v_depth, 1, bias=False)
        self.unproject_qk = torch.nn.Conv2d(self.depth, self.qk_depth, 1, bias=False)
        
    def forward(self, x1, x2, mask=None, x_v_proj=None, dropout_p=0.0):
        batch_size, _, height, width = x1.size()

        v = x1 + x2 # Concatenate the two inputs together
        y = self.project(v) # Apply the project block to the concatenated tensor

        if x_v_proj is None:
          x_v_proj = self.conv(y) # Apply the conv layer to the projection output

        qk = x_v_proj[:, :self.qk_depth] # Split the projection output into q and k
        v = x_v_proj[:, self.qk_depth:]

        if dropout_p > 0.0: # Use dropout if dropout probability > 0.0
            qk = torch.nn.functional.dropout(qk, p=dropout_p) * (1 - mask.float()) # Mask the dropout to avoid using it
        qk = qk.reshape(batch_size, height * width, self.qk_depth) # Reshape the qk output
 
        qk = F.normalize(qk, p=2, dim=2) # Normalize qk by the L2 norm

        w = torch.matmul(qk.transpose(-2, -1), v) # Compute the dot product of the softmax output and the v output
        w = F.normalize(w, p=2, dim=2)
        w = w.reshape(batch_size, self.qk_depth, height, width) # Reshape the softmax output

        w = self.reduce(w) # Apply the reduce block to the softmax output
        v = self.unproject_v(v)
        y = self.unproject_qk(w) # Apply the unproject block to the softmax output
        
        v = v + x2 # Add x2 to the reduced tensor, which is the unprojected output

        if dropout_p > 0.0: # Use dropout if dropout probability > 0.0
            y = torch.nn.functional.dropout(y, p=dropout_p) * (1 - mask.float()) # Mask the dropout to avoid using it
            
        return v, y

class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_probability = dropout_p
        
        self.q_conv = Conv(64,32)
        self.k_conv = Conv(64,32)
        self.v_conv = Conv(64,32)
        self.out_conv = Conv(64,64)
     
    def forward(self, x1, x2, mask=None):
        batch_size, channels, height, width = x1.size()
        mask = mask.float()

        q, y = self.q_conv(x1, x2, mask, dropout_p=self.dropout_probability)
        k, y = self.k_conv(x2, x2, mask, dropout_p=self.dropout_probability)
        v, y = self.v_conv(x2, x2, mask, dropout_p=self.dropout_probability)

        w = torch.matmul(q.transpose(-2, -1), k) # Compute the dot product of the q tensor and the k tensor
        w = F.normalize(w, p=2, dim=3)
        w = w / math.sqrt(channels)
        w = w * mask.float()
        w = torch.nn.functional.dropout(w, p=self.dropout_probability)

        z = torch.matmul(w, v) # Compute the dot product of the softmax output and the value tensor
        z = F.normalize(z, p=2, dim=3)
        z = torch.cat([x1, z], dim=1) # Re-compose the output with input tensor
 
        out = self.out_conv(z, x2, dropout_p=self.dropout_probability) # Apply the residual block to the re-composed output tensor
        return out

# Initializing the model
m = Model(dropout_p=0.0)

# Inputs to the model
