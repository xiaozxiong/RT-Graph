import os
# import numpy as np

os.chdir("../build/bin/")
#! ===== Single Set Intersection =====
#TODO: varing selectivity (0->1, step = 0.1), set size = 0.1M, skew ratio = 1.0, density = 0.01
# for s in range(0, 11, 1):
#     s *= 0.1
#     print(str(s))
#     os.system("./data_generator -l 100000 -r 1 -s "+str(s)+" -d 0.01")

#TODO: varing skew ratio, set size of A = 100K, selectivity = 0.1, density = 0.01, 
# size of B = 10K, 20K, 50K, 100K, 200K, 500K, 1M, 5M, 10M
# for skew_ratio in [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.02, 0.01]:
#     print("size of B = ",100000/skew_ratio)
#     os.system("./s_generator -l 100000 -r "+str(skew_ratio)+" -s 0.1 -d 0.01 -f ../../dataset/skew_ratio/")

#TODO: varing density, set size of A = 10K, size of B = 50K, selectivity = 0.5
# density: 1/100, 1/50, 1/10, 1/5, 1/2, 1/1
# for density in [0.01, 0.02, 0.1, 0.2, 0.5, 1]:
#     print("density = ",density)
#     os.system("./data_generator -l 10000 -r 0.2 -s 0.5 -d "+str(density)+" -f ../../dataset/density/")

#TODO: varing size of A, selectivity = 0.5, skew ratio = 0.8 , density = 0.01, 
#* number of A = 1,000 (default)
# for size in [100, 1000, 10000, 100000, 1000000, 10000000]:
#     print("length of A = ",size)
#     os.system("./s_generator -l "+str(size)+" -r 0.8 -s 0.5 -d 0.01 -f ../../dataset/size/")

#* number of A = 100,000 (test)
# for size in [10, 100, 1000, 10000, 100000, 1000000]:
#     print("length of A = ",size)
#     os.system("./s_generator -a 100000 -l "+str(size)+" -r 0.8 -s 0.5 -d 0.01 -f ../../dataset/size100000/")

#* number of A = 10,000 (test)
# for size in [10, 100, 1000, 10000, 100000]:
#     print("length of A = ",size)
#     os.system("./s_generator -a 10000 -l "+str(size)+" -r 0.8 -s 0.1 -d 0.01 -f ../../dataset/size10000/")

#TODO: vary distribution using exponential distribution
# set selectivity = 0.1, skew = 1, size of A = 0.1 M
# for lamb in [1, 10, 25, 50, 75, 100]:
#     print("lambda = ",lamb)
#     os.system("./s_generator -l 100000 -r 1 -s 0.1 -d 0.1 -b "+str(lamb)+" -f ../../dataset/s_distribution/")
os.system("./s_generator -l 100000 -r 1 -s 0.1 -d 0.1 -b 10 -f ../../dataset/s_distribution/")

#! ===== Multiple Set Intersection =====
#TODO: vary the number of set B
# for n in [1, 10, 100, 1000, 10000, 100000, 1000000]:
#     print("number of set B: ",n)
#     os.system("./m_generator -l 10000 -s 0.1 -n "+str(n)+" -d 0 -f ../../dataset/m_number/")

#TODO: vary selectivity
# for s in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     print("selectivity: ",s)
#     os.system("./m_generator -l 10000 -s "+str(s)+" -n 10000 -d 0 -f ../../dataset/m_selectivity/")

#TODO: vary distribution
# for d in [0, 1, 2]:
#     print("distribution: ",d)
#     os.system("./m_generator -l 10000 -s 0.1 -n 10000 -d "+str(d)+" -f ../../dataset/m_distribution/")
# for lamb in [1, 10, 25, 50, 75, 100]:
#     print("lambda = ",lamb)
#     os.system("./m_generator -l 10000 -s 0.1 -n 10000 -d 2 -f ../../dataset/m_distribution/ -b "+str(lamb))

#TODO: vary skew ratio
# 10K/skew_ratio, selectivity = 0.1, number of sets B = 100
# for skew_ratio in [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.02, 0.01]:
#     print("size of B = ",10000/skew_ratio)
#     os.system("./m_generator -l 10000 -s 0.1 -n 100 -d 0 -r "+str(skew_ratio)+" -f ../../dataset/m_skew/")