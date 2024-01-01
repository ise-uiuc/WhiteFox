
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        # Concatenating x1, x2, and x3 along dimension 0
        t1 = torch.cat([x1, x2, x3], dim=0)

        # Slicing t1 along dimension 1 to only keep the top entries
        # Note the high values
        t2 = t1[:, 0:9223372036854775807]
        
        # Slicing t2 along dimension 1
        # The size of the kept slice is dynamic, thus we pass its size as a parameter
        def size(x):
            return x.shape[1]

        t3 = torch.cat([t2[:, 0, 0, 0, 0], t2[:, size(t2)-1, 0, 0, 0]], dim=0)
        
        # Concatenating the original concatenated tensor t1 and t3 along dimension 1
        t4 = torch.cat([t1, t3], dim=1)

        assert t4.shape[1] == 47 # The static value here is set to 47 to indicate the dynamic value's dimension 1 will be set to 47

        out = t1
        for x in range(t1.shape[0]):
            out[x] *= t1[x]

        assert out.shape[0] == t1.shape[0] # Check that a number of elements in the original tensor remains the same

        # Inverse permutation of t1
        p = torch.tensor([6, 5, 8, 7, 4, 1, 2, 0, 3])
        out = torch.index_select(t1, dim=0, index=p) # Note that 4 in this case will result in an error
        return out

# Initializing the model and passing an input tensor in its initial state
m = Model()

x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(4, 3, 64, 64)
x3 = torch.randn(3, 3, 64, 64)
