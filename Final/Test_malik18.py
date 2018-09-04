"""
This is if a someone imputs some data
python3 Test_malik18.py <Year of Release> <Console> <Critic Score> <# of Critics> <User Scores> <Number of Users> <Genre> <Rating>
python3 Test_malik18.py 2017 PS4 89 115 8.9 5982 Adventure T
"""

import os
import sys

#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

sys.stderr = stderr
np.random.seed(42)

sess=tf.Session() 
saver = tf.train.import_meta_graph('./Saved_Tensor/Until_2016.meta')
saver.restore(sess,tf.train.latest_checkpoint('./Saved_Tensor/'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Out = w2 = graph.get_tensor_by_name("Predicts/kernel:0")

ArrayForIndexing = ['Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score',
       'User_Count', 'Platform_PC', 'Platform_PS3', 'Platform_PS4',
       'Platform_Wii', 'Platform_WiiU', 'Platform_X360', 'Platform_XOne',
       'Genre_Action', 'Genre_Adventure', 'Genre_Fighting', 'Genre_Misc',
       'Genre_Platform', 'Genre_Puzzle', 'Genre_Racing', 'Genre_Role-Playing',
       'Genre_Shooter', 'Genre_Simulation', 'Genre_Sports', 'Genre_Strategy',
       'Rating_E', 'Rating_E10+', 'Rating_M', 'Rating_RP', 'Rating_T']

Year = sys.argv[1]
Console = sys.argv[2]
Critic_Score = sys.argv[3]
Critic_Number = sys.argv[4]
User_Score = sys.argv[5]
User_Number = sys.argv[6]
Genre = sys.argv[7]  
Rating = sys.argv[8]

Input = np.zeros(len(ArrayForIndexing))

Input[0] = int(Year)
Input[1] = float(Critic_Score)
Input[2] = float(Critic_Number)
Input[3] = float(User_Score)
Input[4] = float(User_Number)

for name in ArrayForIndexing:
    if Console in name:
        Input[ArrayForIndexing.index(name)] = 1
    
    if Genre in name:
        Input[ArrayForIndexing.index(name)] = 1
    
    if Rating in name:
        Input[ArrayForIndexing.index(name)] = 1


print(sess.run(Out,{X:Input.reshape([1,len(ArrayForIndexing)])}))
    
    











