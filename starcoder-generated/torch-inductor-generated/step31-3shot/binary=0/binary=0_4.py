
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = tensor([[[[-0.3437,  0.0370, -0.2266],
               [-0.2343,  0.4359,  0.0107],
               [-0.1531,  0.2089,  1.3906]],

              [[ 0.4258, -0.0726, -0.2852],
               [ 0.0086, -0.3048,  0.1763],
               [ 0.0561, -0.1537,  2.2871]],

              [[ 1.6945,  0.0889,  0.0160],
               [ 0.6236,  0.5533, -0.6163],
               [-1.4789, -1.2319, -0.4052]]]])
other = tensor([[[[-0.5458,  0.0136,  0.0286],
                 [-0.0652,  0.0015, -0.4068],
                 [-0.1764,  0.4147,  0.6264]],

                [[ 0.8327, -0.5841,  0.1549],
                 [-0.1427, -0.4024,  0.2850],
                 [-0.5063, -0.3230, -0.0020]],

                [[ 0.2381, -0.4404,  0.2855],
                 [ 0.7881,  1.4808,  0.0637],
                 [-0.7895, -1.4691, -1.3427]]]])
