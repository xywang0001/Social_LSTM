'''
Baseline implementation: https://github.com/ajvirgona/social-lstm-pytorch/tree/master/social_lstm
By Author: Anirudh Vemula
Date: 13th June 2017

Improvements: 1\ Adjust test.py and train.py to accommodate vanilla lstm-> vlstm_train.py, vlstm_test.py
              2\ Adjust optimizer
              3\ reimplementing model.py and vlstm_model.py
              4\ Add Animation.py
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

def visualize(data):
    
    #data: results[i]
    
    '''
    len(results): num_of_batches 

    data[0] : Numpy matrix of shape seq_length x numNodes x 2
    Contains the true trajectories of the nodes

    data[1] : Numpy matrix of shape seq_length x numNodes x 2
    Contains the predicted trajectories of the nodes

    data[2] : A list of lists, of size seq_length
    Each list contains the nodeIDs present at that time-step

    data[3] : Length of observed trajectory
    '''

    anim_running = False
    
    # Get the number of pedestrains in the frame 
    
    max_peds = data[0].shape[1]
    
    # Get the number of frames
    max_frames = data[0].shape[0]
    
    # initialize figure
    fig = plt.figure()
    plt.axis('equal')
    plt.grid()
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black")
    
    # ground truth
    peds_line = []
    peds_dot = []
    
    # prediction
    peds_line_predict = []
    peds_dot_predict = []
    
    #initialize color for different pedestrain
    color = np.random.rand(3, max_peds)
    
    for i in range(max_peds):
        
        temp = ax.plot([], [],'-', lw=2,label = str(i), c = color[:,i])
        peds_line.extend(temp)
        
        temp = ax.plot([], [],'p', lw=2, label=str(i), c=color[:,i])
        peds_dot.extend(temp)
        
        temp = ax.plot([], [],'--', lw=2,label = str(i), c = color[:,i])
        peds_line_predict.extend(temp)
        
        temp = ax.plot([], [],'o', lw=2, label=str(i), c=color[:,i])
        peds_dot_predict.extend(temp)

    fig.subplots_adjust(top=0.8)

    def init():
        
        for ped_line in peds_line:
            ped_line.set_data([], [])
        for ped_dot in peds_dot:
            ped_dot.set_data([], [])
            
        for ped_line in peds_line_predict:
            ped_line.set_data([], [])
        for ped_dot in peds_dot_predict:
            ped_dot.set_data([], [])
            
        return peds_line,peds_dot,peds_line_predict, peds_dot_predict
    

    def animate(i):
        print('frame:', i, 'from: ', max_frames)
        
        ped_list = data[2][i]

        for ped_num, ped_line in enumerate(peds_line):
            
            if ped_num not in ped_list:
                
                                
                ped_line.set_data([], [])
                peds_dot[ped_num].set_data([],[])
                peds_line_predict[ped_num].set_data([], [])
                peds_dot_predict[ped_num].set_data([],[])
                
            else:
                
                (x1,y1) = ped_line.get_data()
                (x2,y2) = peds_line_predict[ped_num].get_data()
                
                ped_line.set_data(np.hstack((x1[:],data[0][i, ped_num, 0])), np.hstack((y1[:],data[0][i, ped_num, 1])))
                peds_line_predict[ped_num].set_data(np.hstack((x2[:],data[1][i, ped_num, 0])), np.hstack((y2[:],data[1][i, ped_num, 1])))
                peds_dot[ped_num].set_data(data[0][i,ped_num,0], data[0][i,ped_num,1])
                peds_dot_predict[ped_num].set_data(data[1][i,ped_num,0], data[1][i,ped_num,1])
                
        return peds_line, peds_dot, peds_line_predict, peds_dot_predict

    # You can pause the animation by clicking on it.
    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True
    
    fig.canvas.mpl_connect('button_press_event', onClick)
    
    # Set up formatting for the movie files
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writer = animation.FFMpegWriter()
    
    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=max_frames,
                                   interval=500)
    
    anim.save('social4.mp4', writer = writer)
    #anim.save('ground_truth_02.html')
    plt.show()
    

def main():
    
    test_dataset = '4'
    save_directory = 'save_social_model/'
    save_directory += str(test_dataset) + '/'
    plot_directory = 'plot/'
    
    f = open(save_directory + 'results3_05.pkl', 'rb')

    results = pickle.load(f)
    size = len(results)
    BATCH_result = results[-1]
    visualize(BATCH_result)
    f.close()
    

if __name__ == '__main__':
    main()