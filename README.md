# supplementary_material_for_2025_paper
- To reproduce the results - download LandCover.ai dataset from https://landcover.ai.linuxpolska.com/ to local directory "/data/landcover.ai/"
- generate patches using the split.py script that comes with the dataset. Set resolution 128 in split.py.
- update params.txt with below

       - For vanila deeplabv3+ with resnet50 backbone : "lcai,resnet50,dlv3,1,0.0,0.0,0.0"
       - For deelabv3+, resnet50 backbone with FSV enabled : "lcai,resnet34,dlv3,1,0.0,0.25,0.25"
- Run train_v2.ipynb
