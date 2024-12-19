import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 创建保存图像的文件夹
output_dir = "deneme"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for j in range(2,11): # 1已经做完了
    patient_num=j

    # 确保每个患者的文件夹存在
    patient_output_dir = os.path.join(output_dir, str(patient_num))
    if not os.path.exists(patient_output_dir):
        os.makedirs(patient_output_dir)

    # peak_loc=np.load("RP"+str(patient_num)+".npy").tolist()
    gan_outputs=sio.loadmat("all_test_outputs/"+str(patient_num)+"/noisy_sig.mat")
    real_sig=sio.loadmat("all_test_outputs/"+str(patient_num)+"/real_sig.mat")
    gan_outputs=gan_outputs["gan_outputs"]
    real_sig=real_sig["real_sig"]

    file_path = os.path.join("R", "R" + str(patient_num) + ".mat")
    ab_beats=sio.loadmat(file_path)
    print(file_path)
    print(ab_beats)
    # ab_beats=sio.loadmat("R"+str(patient_num)+".mat")
    S=ab_beats["S"]
    V=ab_beats["V"]
    print(S,V)

    S = pd.DataFrame(data =S)
    V = pd.DataFrame(data =V)

    win_size=int(np.floor(len(gan_outputs)/4000))

    gan_outputs=gan_outputs[:win_size*4000]

    real_sig=real_sig[:win_size*4000]

    gan_outputs1=gan_outputs.reshape(win_size,4000)

    real_sig1=real_sig.reshape(win_size,4000)

    S_arr=np.zeros(((win_size*4000),1))
    V_arr=np.zeros(((win_size*4000),1))

    S_arr[S.values]=1
    V_arr[V.values]=1
    S_arr=S_arr.reshape(win_size,4000)
    V_arr=V_arr.reshape(win_size,4000)
    # for i in range(0,8000):
    for i in range(0, 8000):

        print(i)
        gan_outputs=gan_outputs1[i,:]


        real_sig=real_sig1[i,:]
        V=V_arr[i,:]
        S=S_arr[i,:]

        # 打印V和S
        print(f"S: {S}")
        print(f"V: {V}")

        time_axis=np.arange(i*4000,(i+1)*4000)/400
        a=plt.figure()
        a.set_size_inches(12, 10)
        ax=plt.subplot(211)
        major_ticksx = np.arange(10*i, 10*(i+1),1 )
        minor_ticksx = np.arange(10*i, 10*(i+1), 0.25)
        major_ticksy = np.arange(-1.5, 1.5,0.3 )
        minor_ticksy = np.arange(-1.5, 1.5, 0.075)
        ax.set_xticks(major_ticksx)
        ax.set_xticks(minor_ticksx, minor=True)
        ax.set_yticks(major_ticksy)
        ax.set_yticks(minor_ticksy, minor=True)

        plt.plot(time_axis,real_sig,linewidth=0.7,color='k')
        plt.scatter(time_axis[real_sig*S!=0], real_sig[real_sig*S!=0], c='#bcbd22',  s=100,marker=(5, 1), alpha=0.5)
        plt.scatter(time_axis[real_sig*V!=0],real_sig[real_sig*V!=0], c='#2ca02c',s=100, marker=(5, 1), alpha=0.5)
        ax.grid(which='minor', alpha=0.2,color='r')
        ax.grid(which='major', alpha=0.5,color='r')
        plt.title("Original ECG Segment", fontsize=15)
        plt.axis([10*i, 10*(i+1),-1.5, 1.5])
        plt.xlabel('Time (seconds)', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        ax2=plt.subplot(212, sharex = ax)
        # Major ticks every 20, minor ticks every 5
        major_ticksx = np.arange(10*i, 10*(i+1),1 )
        minor_ticksx = np.arange(10*i, 10*(i+1), 0.25)
        major_ticksy = np.arange(-1.5, 1.5,0.3 )
        minor_ticksy = np.arange(-1.5, 1.5, 0.075)
        ax2.set_xticks(major_ticksx)
        ax2.set_xticks(minor_ticksx, minor=True)
        ax2.set_yticks(major_ticksy)
        ax2.set_yticks(minor_ticksy, minor=True)
        plt.plot(time_axis,gan_outputs,linewidth=0.7,color='k')
        plt.scatter(time_axis[gan_outputs*S!=0], gan_outputs[gan_outputs*S!=0], c='#bcbd22',  s=100,marker=(5, 1), alpha=0.5)
        plt.scatter(time_axis[gan_outputs*V!=0],gan_outputs[gan_outputs*V!=0], c='#2ca02c',s=100, marker=(5, 1), alpha=0.5)
        ax2.grid(which='minor', alpha=0.2,color='r')
        ax2.grid(which='major', alpha=0.5,color='r')
        plt.title("Operational Cycle-GAN", fontsize=15)
        plt.xlabel('Time (seconds)', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        plt.axis([10*i, 10*(i+1),-1.5, 1.5])

        plt.tight_layout(pad=1.0)
        # fname="deneme/"+str(patient_num)+"/pt_"+str(patient_num)+"_"+str(i)
        fname = os.path.join(patient_output_dir, f"pt_{patient_num}_{i}.png")
        plt.savefig(fname,dpi=200)
        plt.close()



