
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v00 = (1,)
        v01 = (1,)
        v02 = (1,)
        v03 = (1,)
        v04 = (64,)
        v05 = (64,)
        v06 = (64,)
        v07 = (32,)
        v08 = (1,)
        v09 = (1,)
        v10 = (1,)
        v11 = (64,)
        v12 = (1,)
        v13 = (64,)
        v14 = (1,)
        v15 = (32,)
        v16 = (64,)
        v17 = (32,)
        v18 = (3,)
        v19 = (1,)
        v20 = (3,)
        v21 = (1, 1)
        v22 = (0,)
        v23 = x.shape
        v24 = v23[0]
        v25 = v23[2]
        v26 = v23[3]
        v27 = (1,)
        v28 = (0, 0)
        v29 = x[v21:v22:v27].shape
        v30 = v29[0]
        v31 = v29[1]
        v32 = v29[2]
        v33 = v29[3]
        v34 = (v31,)
        v35 = (-1,)
        v36 = (1,)
        v37 = (1,)
        v38 = (3,)
        v39 = torch.max_pool2d(x[v21:v22:v27].pad(v28,v29), v38, v36, v35, v37, v34).shape
        v40 = v39[0]
        v41 = v39[1]
        v42 = v39[2]
        v43 = v39[3]
        v44 = (v41,)
        v45 = (-1,)
        v46 = (1,)
        v47 = (3,)
        v48 = torch.max_pool2d(x[v21:v22:v27].pad(v28,v29), v47, v46, v45, v43, v44).shape
        v49 = v48[0]
        v50 = v48[1]
        v51 = v48[2]
        v52 = v48[3]
        v53 = (v50,)
        v54 = (1,)
        v55 = (-1,)
        v56 = (1,)
        v57 = torch.nn.ModuleDict(items={'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '18':0, '19':0, '20':0, '21':0, '22':0, '23':0, '24':0, '25':0, '26':0, '27':0, '28':0, '29':0, '30':0, '31':0, '32':0, '33':0, '34':0, '35':0, '36':0, '37':0, '38':0, '39':0, '40':0, '41':0, '42':0, '43':0, '44':0, '45':0, '46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '71':0, '72':0, '73':0, '74':0, '75':0, '76':0, '77':0, '78':0, '79':0, '80':0, '81':0, '82':0, '83':0, '84':0, '85':0, '86':0, '87':0, '88':0, '89':0, '90':0, '91':0, '92':0, '93':0, '94':0, '95':0, '96':0, '97':0, '98':0, '99':0})
        v58 = torch.randn(v57, v56, v55, v54, v53).shape
        v59 = v58[0]
        v60 = v58[1]
        v61 = v58[2]
        v62 = v58[3]
        v63 = (v62,)
        v64 = (-1,)
        v65 = (1,)
        v66 = torch.nn.ModuleDict(items={'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '18':0, '19':0, '20':0, '21':0, '22':0, '23':0, '24':0, '25':0, '26':0, '27':0, '28':0, '29':0, '30':0, '31':0, '32':0, '33':0, '34':0, '35':0, '36':0, '37':0, '38':0, '39':0, '40':0, '41':0, '42':0, '43':0, '44':0, '45':0, '46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '71':0, '72':0, '73':0, '74':0, '75':0, '76':0, '77':0, '78':0, '79':0, '80':0, '81':0, '82':0, '83':0, '84':0, '85':0, '86':0, '87':0, '88':0, '89':0, '90':0, '91':0, '92':0, '93':0, '94':0, '95':0, '96':0, '97':0, '98':0, '99':0})
        v67 = torch.randn(v66, v64, v65, v33, v63).shape
        v68 = v67[0]
        v69 = v67[1]
        v70 = v67[2]
        v71 = v67[3]
        v72 = (2, 2)
        v73 = torch.nn.ModuleDict(items={'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '18':0, '19':0, '20':0, '21':0, '22':0, '23':0, '24':0, '25':0, '26':0, '27':0, '28':0, '29':0, '30':0, '31':0, '32':0, '33':0, '34':0, '35':0, '36':0, '37':0, '38':0, '39':0, '40':0, '41':0, '42':0, '43':0, '44':0, '45':0, '46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '71':0, '72':0, '73':0, '74':0, '75':0, '76':0, '77':0, '78':0, '79':0, '80':0, '81':0, '82':0, '83':0, '84':0, '85':0, '86':0, '87':0, '88':0, '89':0, '90':0, '91':0, '92':0, '93':0, '94':0, '95':0, '96':0, '97':0, '98':0, '99':0})
        v74 = torch.randn(v73, v71, v72).shape
        v75 = v74[0]
        v76 = v74[1]
        v77 = (v75, v76)
        v78 = v77[0]
        v79 = v77[1]
        v80 = torch.nn.ModuleDict(items={'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '18':0, '19':0, '20':0, '21':0, '22':0, '23':0, '24':0, '25':0, '26':0, '27':0, '28':0, '29':0, '30':0, '31':0, '32':0, '33':0, '34':0, '35':0, '36':0, '37':0, '38':0, '39':0, '40':0, '41':0, '42':0, '43':0, '44':0, '45':0, '46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '71':0, '72':0, '73':0, '74':0, '75':0, '76':0, '77':0, '78':0, '79':0, '80':0, '81':0, '82':0, '83':0, '84':0, '85':0, '86':0, '87':0, '88':0, '89':0, '90':0, '91':0, '92':0, '93':0, '94':0, '95':0, '96':0, '97':0, '98':0, '99':0})
        v81 = torch.randn(v80, v79, v78).shape
        v82 = v81[0]
        v83 = v81[1]
        v84 = v81[2]
        v85 = (32,)
        v86 = torch.randn(v84, v83, v82)
        return v86
# Inputs to the model
x = torch.randn(32, 2048, 1, 1)
