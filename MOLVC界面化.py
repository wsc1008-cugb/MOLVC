import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sys
import os
import shutil
import glob
import random
import math
from numpy import random
import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import pandas
from collections import Counter
from matplotlib.font_manager import FontProperties  # 步骤一，进行中文字体的显示但不改变现有英文字体
from matplotlib import ticker #定义百分比坐标
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.decomposition.asf import ASF
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination import get_termination
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import rpy2.robjects as robjects
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
from matplotlib.pyplot import MultipleLocator
from itertools import product
print(sys.version)
print('numba version: {}'.format(np.__version__))
print('flopy version: {}'.format(flopy.__version__))

'''
定义读取mppth函数
'''
def save_mp_resnb(pth_file, out_csv, tar_colnames=None, add_ij_info=False, in_model=None):
    if tar_colnames == None:
        tar_colnames = ["particleid", "x", "y", "z","time", "k"]
    pthobj = flopy.utils.PathlineFile(pth_file)
    df_all = pandas.DataFrame(pthobj._data)[tar_colnames]
    if add_ij_info:
        i,j = in_model.dis.get_rc_from_node_coordinates(df_all["x"].values, df_all["y"].values)
        df_all.insert(len(df_all.columns), "i", i)
        df_all.insert(len(df_all.columns), "j", j)
    df_all.to_csv(out_csv, index=None)
    return "Complete: %s"%(pth_file)

dir_path=r"D:/工作需/R程序计算/R程序翻译" #路径1
dir_path2=r"D:/工作需/R程序计算/R程序翻译/C13-test" #路径2
timezlh=dir_path+"/timeC13.txt" #文件路径

zlc_dir=dir_path+"/总流场"
dylc_dir=dir_path+"/单元流场"

jzb_dir=dir_path+"/井坐标文件" #井坐标文件

jzb_file_lis=os.listdir(jzb_dir)
if jzb_file_lis[0].startswith("抽井"):
    cj_path=jzb_dir+"/"+jzb_file_lis[0]
    zj_path=jzb_dir+"/"+jzb_file_lis[1]
else:
    cj_path = jzb_dir + "/" + jzb_file_lis[1]
    zj_path = jzb_dir + "/" + jzb_file_lis[0]
    
def intjudge(x):
    """判断粒子数目是否为整数"""
    if  isinstance(x, int)==True:
        index=-1
    else:
        index=1
    return index
 
def newly_build_dir():
    """ 检测保存路径是否存在，不存在则创建"""
    for i in (zlc_dir,dylc_dir):
        if not os.path.exists(i):
            os.mkdir(i)

def get_file_list(inwell):
    """ 获取要处理文件的列表"""
    with open(timezlh,"r",encoding="utf-8") as f:
        l=f.read().strip().split("\n")
    file_list=[(i,dir_path2+"/"+"cell_"+i+".txt",dir_path2+"/"+"in_1800_%d.txt"%(int(inwell)/2+1)) for i in l]
    return file_list

def get_cj_ijk():
    """ 获取抽井ijk"""
    data=pandas.read_csv(cj_path,sep="\\s+",usecols=["I","J","K"]).values.tolist() #读取ijk三列并转换为列表
    set_data=set([tuple(i) for i in data]) #将i,j,k放入元组并使用集合去重
    return set_data

def get_zj_ijk(file_path):
    """ 获取注井ijk"""
    dic={}
    data = pandas.read_table(zj_path, sep="\\s+").values.tolist()
    for i in data:
        dic.setdefault(i[-1],set()).add(tuple(i[:3]))
    for i in dic:
        dic[i]=tuple(dic[i])
    dic1={}
    data = pandas.read_table(file_path, sep="\\s+", usecols=["Particle_Index","Time", "Cell_K", "Cell_I", "Cell_J"])
    data=data[["Particle_Index","Time", "Cell_I", "Cell_J", "Cell_K"]]
    data["ijk"]=data.apply(lambda x:(x["Cell_I"],x["Cell_J"],x["Cell_K"]),axis=1)
    for k,v in dic.items():
        dic1[k]=set()
        all_index=[]
        for i in v:
            p=data[(data["ijk"]==i) & (data["Time"]==0)]
            l=p["Particle_Index"].values.tolist()
            all_index+=l
        data1=data[data["Particle_Index"].isin(all_index)][["Cell_I", "Cell_J", "Cell_K"]].values.tolist()
        for i in data1:
            dic1[k].add(tuple(i))
    return dic1

def get_set_A(file_path):
    """ 获取集合A"""
    cj_ijk=get_cj_ijk()
    data = pandas.read_table(file_path,sep="\\s+", usecols=["Particle_Index","Time","Cell_K","Cell_I","Cell_J"])
    data["ijk"] = data.apply(lambda x: (x["Cell_I"], x["Cell_J"], x["Cell_K"]), axis=1)
    data1=data[data["ijk"].isin(cj_ijk)]["Particle_Index"].values.tolist()
    d=tuple(set(data1))
    data=data[(data["Time"]==0) & (data["Particle_Index"].isin(d))][["Cell_I","Cell_J","Cell_K"]].values.tolist()
    set_data = set([tuple(i) for i in data])  # 将i,j,k放入元组并使用集合去重
    set_A=set_data
    return set_A

def get_set_B(file_path):
    """ 获取集合B"""
    data = pandas.read_table(file_path, sep="\\s+", usecols=["Time", "Cell_K", "Cell_I", "Cell_J"])
    data = data[["Cell_I", "Cell_J", "Cell_K"]].values.tolist()
    set_data = set([tuple(i) for i in data])  # 将i,j,k放入元组并使用集合去重.元组中自动去重
    return set_data

def save_csv(csv_path,s):
    """将集合保存为csv文件 """
    data=pandas.DataFrame(data=tuple(s),columns=["Cell_I", "Cell_J", "Cell_K"])
    data.to_csv(csv_path,index=False)

def zlc_save(d,j,b,c,set_A,set_B):
    """ 保存总流场计算结果"""
    d=zlc_dir+"/"+d
    if not os.path.exists(d):
        os.mkdir(d)
    save_csv(d+"/有效对流流场.csv",j)
    save_csv(d+"/总对流流场.csv",b)
    save_csv(d+"/无效对流流场.csv",c)
    save_csv(d+"/set_A.csv",set_A)
    save_csv(d+"/set_B.csv",set_B)
    
def cal_Lv(data_mine,carr):
    Vsum=0
    for i in range(carr.shape[0]):
        for j in range(data_mine.shape[0]):
            if data_mine[j, 1] == carr[i,1] and data_mine[j, 2] == carr[i,2] and data_mine[j, 0] == carr[i,0]:
                Vsum += 62.109
                break
            else:
                continue
    return Vsum

def calculation_dylc(zj_ijk,set_A,n):
    """ 计算单元流场"""
    save_dic=dylc_dir+"/"+n
    if not os.path.exists(save_dic):
        os.mkdir(save_dic)
    for k,v in zj_ijk.items():
        d=save_dic+"/"+k
        if not os.path.exists(d):
            os.mkdir(d)
        set_jiao = set_A & v  # 交集
        set_bing = set_A | v  # 并集
        set_cha = (set_A - v) | (v-set_A)  # 差集
        save_csv(d + "/有效对流流场.csv", set_jiao)
        save_csv(d + "/总对流流场.csv", set_bing)
        save_csv(d + "/无效对流流场.csv", set_cha)
        
def mpthin(ptcl,inwell):
    workspace = os.path.join('.')
    time=[]
    with open(r'D:\\工作需\\R程序计算\\R程序翻译\\timeC13.txt', 'r+') as obj:
        for line in obj.readlines():
            time.append(line)
        time = [line.strip("\n") for line in time]
    Tlist=np.array(time)
    #定义模型名称和存储文件名
    nm='C13-test'
    '''定义模板'''
    is_silent = True  #是否隐藏模型运行及modpath的完整消息
    in_mfn=r'E:\G-case/C13-test_MODFLOW_text/C13-test.mfn' #修改需导入的.mfn文件路径
    m = flopy.modflow.Modflow.load(in_mfn, model_ws=workspace, version="mf2005", exe_name = 'mf2005.exe')
    nrow, ncol, nlay, nper = m.nrow_ncol_nlay_nper
    m.write_input()
    m.run_model(silent=is_silent)
    #在所有in内加入粒子
    data = pandas.read_table(r"D:\工作需\R程序计算\R程序翻译\井坐标文件\注井坐标C13.txt", sep="\\s+").values.tolist()
    In_wel=np.array(data)
    inw=In_wel[inwell:inwell+2,:]
    plocs = []
    pids = []
    for i in inw:
        plocs.append((int(i[2])-1,int(i[0])-1,int(i[1])-1))
    Plocs=np.concatenate([plocs, plocs], axis=1)
    cd = flopy.modpath.CellDataType(drape=0,
                                   columncelldivisions=ptcl[0],
                                   rowcelldivisions=ptcl[1],
                                   layercelldivisions=ptcl[2])
    p = flopy.modpath.LRCParticleData([cd], [Plocs])

    pg1 = flopy.modpath.ParticleGroupLRCTemplate(particlegroupname="PG2",
                                            particledata=p,
                                            filename="ex01a.pg2.sloc")
    particlegroupin = [pg1] #考虑设置一系列井的粒子
    import time
    for idx,i in enumerate(Tlist):
        s = time.time()
        # create modpath files
        exe_name = "mp7"
        mp = flopy.modpath.Modpath7(
            modelname=nm + "_mp", flowmodel=m, exe_name=exe_name, model_ws=workspace, budgetfilename="C13-test.ccf"
        )
        mpbas = flopy.modpath.Modpath7Bas(mp)
        mpsim = flopy.modpath.Modpath7Sim(
            mp,
            simulationtype="pathline",
            trackingdirection="forward",
            weaksinkoption="pass_through",
            weaksourceoption="pass_through",
            budgetoutputoption="summary",
            referencetime=[0, 0, 0.0], #起始时间点
            stoptimeoption="specified",
            stoptime=int(i),
            zonedataoption="off",
            particlegroups=particlegroupin,
        )

        # write modpath datasets
        mp.write_input()
        # run modpath
        mp.run_model(silent=True)
        '''加入I J并输出'''
        fpth_file = workspace+"/"+nm+"_mp.mppth"
        out_csv = workspace+"/modpath_midresult.csv"
        '''调用函数进行处理'''
        save_mp_resnb(fpth_file, out_csv, add_ij_info=True, in_model=m)

        df=pandas.read_csv(out_csv) #默认第一行为标题
        c=df.values.tolist()
        carr=np.array(c, dtype='object')
        rows, columns = carr.shape
        columns=['Particle_Index', 'X', 'Y', 'Z', 'Time', 'Cell_K', 'Cell_I', 'Cell_J']
        for i,j in product(range(rows),range(5,8)): #实现扁平化循环
            carr[i,j]=int(carr[i,j]+1)
            carr[i,0]=int(carr[i,0])
        carr=np.insert(carr, 0, values=columns, axis=0)
        '''定义结果存储位置'''
        txtc=[]
        for i in Tlist:
            txtcell = os.path.join(r"D:\工作需\R程序计算\R程序翻译\%s"%nm, "in_1800_%d.txt"%(int(inwell)/2+1))
            txtc.append(txtcell)
        wstxt=r"D:/工作需/R程序计算/R程序翻译/"+nm
        if not os.path.exists(wstxt):
            os.makedirs(wstxt)
        with open(txtc[idx], 'w') as f: #保存至指定文件
            np.savetxt(f, carr, fmt="%s", delimiter="    ")
        e = time.time()
        print("%s/%s | %s completed in %.2fs"%(idx+1, len(Tlist), txtc[idx], e-s))
mine = pandas.read_table(r'D:\工作需\R程序计算\R程序翻译\0222矿层.txt', sep="\\s+").values.tolist()
arrm=np.array(mine)
class MyProblem(ElementwiseProblem):
    def __init__(self, inwell,ll, ul, volsim):
        super().__init__(n_var=3,              # 变量数，注井在三个方向投放的粒子数
                            n_obj=2,              # 目标数，最大化浸染面积和最小化投放粒子数
                            n_ieq_constr=1,       # 约束条件数：每个方向最少有2个粒子，最多有10个粒子；
                            xl=np.array([ll, ll, ll]),            # 下限
                            xu=np.array([ul, ul, ul]),            # 上限
                            vtype=int)  
        
        self.inwell=inwell
        self.volsim=volsim

    def _evaluate(self, x, out, *args, **kwargs):
        vol = self.calculation(x)

        '''f为目标值'''
        f1 = x[0]*x[1]*x[2]        # 最小化粒子总数；x是长度为n_var的一维数组
        f2 = -vol                  # 最大化有效对流体积

        '''g为约束条件转成求 <= 0'''
        g1=x[0]*x[1]*x[2]-2000

        out["F"] = [f1, f2] # 目标值，转成求最小值
        out["G"] = [g1] #约束条件
    def calculation(self, ptcl):
        """ 计算结果并保存"""
        #第i/2+1注井
        newly_build_dir()
        mpthin(ptcl,self.inwell)      #先执行IN的流场模拟
        for i in get_file_list(self.inwell):
            set_A=get_set_A(i[1])
            set_B=get_set_B(i[2])
            set_jiao=set_A & set_B #交集
            arr_real=np.array([i for i in set_jiao])
            Vm=cal_Lv(arrm,arr_real)
            set_bing=set_A | set_B #并集
            set_cha=(set_A - set_B) | (set_B - set_A) #差集
            zlc_save(i[0],set_jiao,set_bing,set_cha,set_A,set_B)
            volume=len(set_jiao)*self.volsim
        return Vm 

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi")
        self.geometry("1000x700")
        # 创建左右两个Frame
        left_frame = tk.Frame(self)
        left_frame.grid(row=0, column=0, padx=10, pady=10)

        right_frame = tk.Frame(self)
        right_frame.grid(row=0, column=1, padx=10, pady=10)

        # 创建标签和文本框，并分别添加到左右两个Frame中
        self.label_inwell = ttk.Label(left_frame, text="注井号:")
        self.label_inwell.grid(row=1, column=0, pady=5)
        self.entry_inwell = ttk.Entry(right_frame)
        self.entry_inwell.grid(row=1, column=0, pady=5)

        self.label_ll = ttk.Label(left_frame, text="单井粒子数下限:")
        self.label_ll.grid(row=2, column=0, pady=5)
        self.entry_ll = ttk.Entry(right_frame)
        self.entry_ll.grid(row=2, column=0, pady=5)

        self.label_ul = ttk.Label(left_frame, text="单井粒子数上限:")
        self.label_ul.grid(row=3, column=0, pady=5)
        self.entry_ul = ttk.Entry(right_frame)
        self.entry_ul.grid(row=3, column=0, pady=5)

        self.label_volsim = ttk.Label(left_frame, text="单元格体积:")
        self.label_volsim.grid(row=4, column=0, pady=5)
        self.entry_volsim = ttk.Entry(right_frame)
        self.entry_volsim.grid(row=4, column=0, pady=5)

        # 创建按钮
        self.button_run = ttk.Button(self, text="运行", command=self.run)
        self.button_run.grid(row=1, column=0, columnspan=2, pady=10)

        #创建画布
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('number', )
        self.ax.set_ylabel('volume')
        self.ax.set_title('MOLVC Result')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # 让窗口自适应大小，自定义尺寸
        self.resizable(False, False)
        self.update()
        
        
    def run(self):
        inwell = int(self.entry_inwell.get())
        ll = int(self.entry_ll.get())
        ul = int(self.entry_ul.get())
        volsim = self.entry_volsim.get()
        
        #输入参数验证
        if not all([inwell, ll, ul, volsim]):
            messagebox.showerror("错误", "请输入所有参数！")
            return

        try:
            inwell = int(inwell)
            ll = int(ll)
            ul = int(ul)
            volsim = float(volsim)
        except ValueError:
            messagebox.showerror("错误", "请输入正确的参数格式！")
            return
        
        #计算过程
        problem = MyProblem(inwell, ll,ul,volsim)
        algorithm = NSGA2(
            pop_size=2,  #种群规模（初代数量）
            n_offsprings=10, #之后每一代的后代数目（叠加计算次数）
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.7, eta=15),
            mutation=PM(prob=0.05, eta=20),
            eliminate_duplicates = True
            )
        '''定义终止原则'''
        termination = get_termination("n_gen", 2) #迭代计算  
        '''执行多目标优化程序进程'''
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        X = res.X # 变量
        F = res.F # 目标    
        print(X)
        print(F)
        arrF=np.array(F)
        from sklearn.preprocessing import MinMaxScaler
        nF = MinMaxScaler().fit_transform(arrF) 
        '''一些注释需要注意的'''
        #默认的feather_range为缩放到0-1之间，F中是包含f1和f2两个结果的
        # nF = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))   
        '''熵权法确定两个目标的权重'''
        def Entropy(data):    
            P_ij = data / data.sum(axis=0)
            e_ij = (-1 / np.log(data.shape[0])) * P_ij * np.log(P_ij)
            e_ij = np.where(np.isnan(e_ij), 0.0, e_ij)
            return (1 - e_ij.sum(axis=0)) / (1 - e_ij.sum(axis=0)).sum()

        weights = np.array([0.2,0.8])
        '''增强的标量化函数'''
        decomp = ASF()
        i = decomp.do(nF, 1/weights).argmin()

        print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))
        print(X[i])
if __name__ == "__main__":
    app = Application()
    app.mainloop()
