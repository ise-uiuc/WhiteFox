
class Model(torch.nn.Module):
    def forward(self, X1, X2, X3, X4):
        v1 = X1.reshape((-1, 196)).mm(X2.reshape((196, -1)))
        v2 = v1 / (-2.2909912153930664 + 1.0647775173187256)
        v3 = v2[:536, :26]
        return X3.reshape((-1, 384)).mm(X4.reshape((384, -1))).mm(v3)
   
# Initializing the model
m = Model()

# Inputs to the model
X1 = torch.randn(17, 196)
X2 = torch.randn(196, 26)
X3 = torch.randn(1, 384)
X4 = torch.randn(384, 26)
