:
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 3)
        self.fc2 = torch.nn.Linear(4, 5)
        self.fc3 = torch.nn.Linear(2, 2, bias=False)
    def forward(self, input):
        out = torch.tanh(self.fc1(input))
        v16 = []
        v21 = []
        v11 = out.numpy()
        v11 = list(v11)
        v27 = dict()
        for v28 in self.fc2.state_dict().items() :
            v27[v28[0]] = v28[1]
        for i in range(len(v11)):
            v102 =[]
            v101 =[]
            v111 = v11[i]

            v101.append(v111)
            v101[0] = v101[0].tolist()
            v101[0][0] = float(v101[0][0])

            v101.append(self.fc2.weight)
            v101[1] = v101[1].tolist()
            v101[1][0] = v101[1][0].tolist()
            v101[1][0][0] = v101[1][0][0].tolist()
            v10 = v101[0][1][0][0]
            v10 = torch.tensor(v10)
            v1 = v101[1][0]
            v11 = v1 * v10
            v11 = v11.tolist()
            v110 = v11[0]
            v9 = v110[0]
            v8 = v9 + v27['bias']
            v8 = v8.tolist()
            v27['bias'] = v8
            v79 = dict()
            for v80 in v27.items() :
                v79[v80[0]] = v80[1]

            v78 = list()
            for v81 in list(v110):
                v82 = v81
                v77 = dict()
                for v83 in v27.items() :
                    v77[v83[0]] = v83[1]

                v76 = v82
                for key in list(v77.keys()):
                    v84 = v77[key]
                    v85 = v84
                    v86 = v85 * v76
                    v76 = v86
                v78.append(v76)
            v103 = v78

            v99 =[]
            v98=[]
            v75 = v103[0]
            v112 = v75.item()
            v100 = v112
            v99.append(v100)
            v25 = v79['bias']
            v100 = v25[0]
            v113 = v100.item()
            v100 = v113
            v99.append(v100)
            v26 = v79['weight']
            v100 = v26[0]
            v114 = v100.item()
            v100 = v114
            v99.append(v100)
            v27 = v79['weight']
            v100 = v27[1]
            v115 = v100.item()
            v100 = v115
            v99.append(v100)
            v13 = v103[1]
            v116 = v13.item()
            v100 = v116
            v99.append(v100)
            v14 = v103[2]
            v117 = v14.item()
            v100 = v117
            v99.append(v100)
            v17 = v103[3]
            v118 = v17.item()
            v100 = v118
            v99.append(v100)
            v15 = v103[4]
            v119 = v15.item()
            v100 = v119
            v99.append(v100)
            v19 = v103[5]
            v120 = v19.item()
            v100 = v120
            v99.append(v100)
            v16.append(v99)
            v21.append(v16)
        v74 = v21[0][0]
        v6 = v74
        v7 = v6[0]
        v23 = v7
        v22 = torch.tensor(v23)
        v11 = v21[0][0]
        v24 = v11[1]
        v5 = v24
        v3 = v5
        v69 = v16[0][1]

        #v69 = 0
        #v69 = 1
        return v69
