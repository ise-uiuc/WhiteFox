
class ModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(10, 6, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = v1 * 0.69314718055994530941723212145818
        v3 = v2 + 0.91629073187415500084525997942612
        v4 = torch.asin(v3)
        v5 = v4 * 0.77880078307140445412288705827529
        v6 = torch.sinh(v4)
        v7 = torch.exp(v2)
        v8 = v6 + 0.63640146895494185455681797300204
        v9 = v7 + 0.85807848702670765048625994381545
        return v9
class ModelB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(6, 12, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = v1 * 0.30252684415302588600758395078069
        v3 = v2 + 0.69314718055994530941723212145818
        v4 = torch.asin(v3)
        v5 = v4 * 0.82246268320413651012220430786351
        v6 = torch.sinh(v4)
        v7 = torch.exp(v2)
        v8 = v6 + 0.91192710955915098263247649695853
        v9 = v7 + 0.97534446640530979081848872945491
        return v9
class ModelC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(12, 20, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = v1 * 0.51082562376599068153976625293059
        v3 = v2 + 0.69314718055994530941723212145818
        v4 = torch.asin(v3)
        v5 = v4 * 0.89616884826277110522127389202826
        v6 = torch.sinh(v4)
        v7 = torch.exp(v2)
        v8 = v6 + 1.0986089809758952922004953258862
        v9 = v7 + 1.0218674335299377573488996462376
        return v9
class ModelD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(20, 23, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.acos(v1)
        v3 = torch.cosh(v1)
        v4 = torch.exp(v1)
        v5 = v3 + 0.69314718055994530941723212145818
        v6 = v4 + 0.87445126861529562781036309438920
        v7 = torch.tanh(v2)
        v8 = torch.exp(v1)
        v9 = v7 + 1.0986089809758952922004953258862
        v10 = v8 + 1.3132616875182228813261886146198
        return v10
class ModelE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(10, 16, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = v1 * 0.3025809263930141
        v3 = v1 * 0.887558665268145
        v4 = torch.asin(v3)
        v5 = v1 - 0.69314718055994530941723212145818
        v6 = torch.sinh(v1)
        v7 = torch.exp(v3)
        v8 = v4 * 0.530919307325527
        v9 = v4 * 0.805914723098916
        return v10
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.modelA = ModelA()
        self.modelB = ModelB()
        self.modelC = ModelC()
        self.modelD = ModelD()
        self.modelE = ModelE()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(24, 23, 1, stride=1)
    def forward(self, x1):
        v1 = self.modelA(x1)
        v2 = self.modelB(x1)
        v3 = self.modelC(x1)
        v4 = self.modelD(x1)
        v5 = self.modelE(x1)
        v6 = v4 + v5
        v7 = v1 * v6
        v8 = v2 * v6
        v9 = v3 + v4
        v10 = v7 + v9
        v11 = v8 + v9
        v12 = v5 + v6
        v13 = v10 + v12
        v14 = v11 + v12
        v15 = v10 * v11
        v16 = v14 + v15
        v17 = v14 * v12
        v18 = v17 + v15
        v19 = v15 + v16
        v20 = v16 * v17
        v21 = v18 + v19
        v22 = v17 * v18
        v23 = v18 + v21
        v24 = v16 * v19
        v25 = v24 + v23
        v26 = v2 * v3
        v27 = v22 + v24
        v28 = v26 * v25
        v29 = v26 + v27
        v30 = v22 * v28
        v31 = v26 + v28
        v32 = v22 * v30
        v33 = v30 + v31
        v34 = v28 * v30
        v35 = v30 * v32
        v36 = v33 + v34
        v37 = v34 + v35
        v38 = v35 + v37
        v39 = v36 * v37
        v40 = v36 * v38
        v41 = v34 * v40
        v42 = v38 * v41
        v43 = v36 * v42
        v44 = v38 * v40
        v45 = v32 * v38
        v46 = v45 + v44
        v47 = v45 * v43
        v48 = v46 + v47
# Inputs to the model
x1 = torch.randn(1, 10, 15, 15)
