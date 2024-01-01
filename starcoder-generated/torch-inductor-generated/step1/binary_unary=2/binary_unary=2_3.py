
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = torch.nn.Parameter(other.reshape(1, 1, 1, 1))
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.other
        v3 = v2 - v1
        v4 = np.array([[-1.027756e+00, -2.286639e-03, -1.510427e-03, -9.867046e-04]
                      [-4.274235e-03, -1.359701e-02, -1.104885e-02, -6.294370e-03]
                      [-9.072330e-08, 7.456594e-08, 1.389359e-02, 2.382906e-03]
                      [3.021077e-07, -2.305938e-07, 6.808552e-03, 7.897332e-04]
                      [5.815360e-03, 1.292148e-02, 2.338480e-01, 8.578209e-01]
                      [1.520595e-02, 3.752044e-03, 7.897332e-04, 1.060929e-01]
                      [-1.104885e-02, -6.238518e-03, -1.636302e-02, -5.617751e-03]
                      [-4.738528e-03, -9.018803e-03, -2.910816e-01, -6.505371e-01]]).reshape(1, 1, 8, 8)
        v4 = torch.from_numpy(v4)
        v5 = v3 + v4
        v6 = torch.nn.ReLU()(v5)
        return v6

# Initializing the model
other = np.array([-0.12918424, -0.21271498, -0.06773838, 0.04655406]).reshape(1, 4, 1, 1)
m = Model(other)

# Inputs to the model
x = torch.tensor(5.20944614e-01).reshape(1, 3, 8, 8)
