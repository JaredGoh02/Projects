import os
for i in range(1,9001):
    file_name = "image_%d.png" %(i)
    os.rename("C:/Users/USER/OneDrive - Monash University/Desktop/Year4 sem2/ECE4078/ECE4078-Thursday-12-3-G6/Milestone3/Image/superimpose_image/" + file_name,"C:/Users/USER/OneDrive - Monash University/Desktop/Year4 sem2/ECE4078/ECE4078-Thursday-12-3-G6/Milestone3/network/scripts/dataset/images/" + file_name)