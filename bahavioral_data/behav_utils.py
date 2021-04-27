%matplotlib qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from glob import glob
import shutil

# -------------------------------------------------------------------------
def con_subj(): # this function create IQR filtered file of each subject 
    try:
        #elimination subject number 7 and 11 relating to eeg detect data
        #subjecs of 1,3,4,5,6,8,9,10,12,13,14,15,16,17 are remainded
        ds_root = 'PDM_data\\\raw'
        anlz_dir = 'PDM_data\\filtered_subj'
        out_MR= 'sourcedata-eeg_outside-MRT\\beh';
        in_MR = 'sourcedata-eeg_inside-MRT\\beh';
        #remove the analyze_subj folder if it exsists 
        if os.path.exists(anlz_dir):
            shutil.rmtree(anlz_dir)
            
        # preprocess behavioral data from outside the scanner
        list_out_subj = ['sub-001', 
                         'sub-003', 
                         'sub-004', 
                         'sub-005', 
                         'sub-006', 
                         'sub-008', 
                         'sub-009', 
                         'sub-010', 
                         'sub-012',
                         'sub-013',
                         'sub-014',
                         'sub-015', 
                         'sub-016',
                         'sub-017']
                
        list_subj = []
        list_subj = list_out_subj
        folder_MR  = out_MR
            
        for subj in list_subj:
            # subject source and target directories        
            src_dir_subj = os.path.join(ds_root, subj,folder_MR)
            src_dir_subj = os.path.normpath(src_dir_subj)
            tgt_dir_subj = os.path.join(anlz_dir, subj, side)
            tgt_dir_subj = os.path.normpath(tgt_dir_subj)

            if not os.path.exists(tgt_dir_subj):
                os.makedirs(tgt_dir_subj)

            #define and concatenate all file for each subject
            data = pd.DataFrame({'condition' : [],
                      'image_index' : [],
                      'key_press' : [],
                      'response_time' : [],
                      'response_corr' : [],
                      'stimulus_side' : [],
                      'prioritization_cue' : []})           

            #read two beh files for each subject
            for subj_beh in glob(src_dir_subj + "\\*.tsv"):
                #read tsv file
                df = pd.read_csv(subj_beh, sep='\t') 
                #concatenate two file for each dubject
                data = pd.concat([data,df], sort=False)               
                    
            #filter with Boxplot
            Q1 = data['response_time'].quantile(0.25)
            Q3 = data['response_time'].quantile(0.75)
            IQR = Q3 - Q1    #IQR is interquartile range.
            filterr = (data['response_time'] >= Q1 - 3 * IQR) & (data['response_time'] <= Q3 + 3*IQR)
            data = data.loc[filterr]
              
            # Create a Pandas Excel writer using XlsxWriter as the engine.
            save_file = os.path.join(tgt_dir_subj,subj)
             
            #create save_file folder 
            if not os.path.exists(tgt_dir_subj):
                os.makedirs(tgt_dir_subj)
            # save csv dataframe
            data.to_csv(save_file)

            print(subj + ' is successfully saved')      
            
    except Exception as inst:
        print(inst)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------    
# create on csv file for all filtered subjects
def concat_group():
    try:
        # analyze group behavioural data . 
        anlz_dir = 'PDM_data\\filtered_subj'
        anlz_group ='PDM_data\\filtered_group' 
        #remove the analyze_goup folder if it exsists 
        if os.path.exists(anlz_group):
            shutil.rmtree(anlz_group)
       # preprocess behavioral data from outside the scanner
        list_out_subj = ['sub-001', 
                         'sub-003', 
                         'sub-004', 
                         'sub-005', 
                         'sub-006', 
                         'sub-007', 
                         'sub-008', 
                         'sub-009', 
                         'sub-010', 
                         'sub-011', 
                         'sub-012',
                         'sub-013',
                         'sub-014',
                         'sub-015', 
                         'sub-016',
                         'sub-017']        
   
        switcher = {
                'sub-001':1,
                'sub-002':2,
                'sub-003':3,
                'sub-004':4,
                'sub-005':5,
                'sub-006':6,
                'sub-007':7,
                'sub-008':8,
                'sub-009':9,
                'sub-010':10,
                'sub-011':11,
                'sub-012':12,
                'sub-013':13,
                'sub-014':14,
                'sub-015':15,
                'sub-016':16,
                'sub-017':17      
            }
            
        
        list_subj = list_out_subj                  
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        save_file = os.path.join(anlz_group, side)
        #create save_file folder 
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        
        for subj_dir in list_subj:
            tgt_dir_subj = os.path.join(anlz_dir, subj_dir, side, subj_dir)
            tgt_dir_subj = os.path.normpath(tgt_dir_subj)
            
            df_data = pd.read_excel(tgt_dir_subj)
            
            #number of subject 
            subj_idx=np.zeros(len(df_data))
            print(len(df_data))
            #type of stimulus face = 2, car =1
            stim=np.zeros(len(df_data))
            #response to stimulus face = 1, car =0
            response = df_data['key_press'].to_numpy()
            rt=df_data['response_time'].to_numpy()
            #condition
            condition = df_data['condition'].to_numpy()
            #response error
            response_corr = df_data['response_corr'].to_numpy()
            #image_index
            image_index = df_data['image_index'].to_numpy()
            for i in range(len(df_data)):
                subj_idx[i] = switcher.get(subj_dir)
                if(response_corr[i]==0):
                    if(response[i]==2):
                        stim[i] = 1
                    elif(response[i]==1):
                        stim[i] = 2
                else:
                    stim[i]= response[i]
            
            df_new = pd.DataFrame(np.column_stack([subj_idx, stim, response, rt, condition, response_corr, image_index]),
                                  columns=['subj_idx','stimulus','response','rt','condition','response_corr', 'image_index'])
            
            
            #remoce empty filed
            df_new = df_new.loc[np.invert(df_new['rt'].isnull())]
            df_data = pd.concat([df_data, df_allRuns_all_new], sort=False)

        df_data.to_csv(save_file)
        print(' group is successfully saved')
        
    except Exception as inst:
        print(inst)
# -------------------------------------------------------------------------    


# -------------------------------------------------------------------------    
#plots bahavioral data ana rt distribution 
def performance_analysis():
    data = pd.read_csv('PDM_data\\\filtered_group\\outside\\group_outside.csv')
    #elimination subject number 7 and 11 relating to eeg detect data
    #data = data[(data['subj_idx']!=7)&(data['subj_idx']!=11)]

    data['coherency'] = data['condition']
    data['spatial'] = data['condition']
        
    #just spatial condition
    data['spatial']=data['spatial'].replace(1,'Yes')
    data['spatial']=data['spatial'].replace(2,'No')
    data['spatial']=data['spatial'].replace(3,"Yes")
    data['spatial']=data['spatial'].replace(4,'No')
        
    #just coherency condition
    data['coherency']=data['coherency'].replace(1,'High')
    data['coherency']=data['coherency'].replace(2,'High')
    data['coherency']=data['coherency'].replace(3,"Low")
    data['coherency']=data['coherency'].replace(4,'Low')

    #face 1
    #car 0
    #rersponse
    data['response']=data['response'].replace(1,0)
    data['response']=data['response'].replace(2,1)

    #stimulus
    data['stimulus']=data['stimulus'].replace(1,0)
    data['stimulus']=data['stimulus'].replace(2,1)

    #create dataframe to collect rt and acc for all subject
    performance_data = pd.DataFrame({'ACC' : [],
                                     'RT' : [],
                                     'Condition' : [],
                                     'Subject' : [],
                                     'Stimulus' : []}) 

        
    for i in range(1,18):
        if i!=2 and i!=7 and i!=11:
            select_subject =i
            data1 = data[data['subj_idx']==select_subject]
            #data for condition
            data_High_Yes_1 = data1[(data1['coherency']=='High')&(data1['spatial']=='Yes')&(data1['stimulus']==1)]
            data_High_Yes_0 = data1[(data1['coherency']=='High')&(data1['spatial']=='Yes')&(data1['stimulus']==0)]
            data_High_No_1  = data1[(data1['coherency']=='High')&(data1['spatial']=='No')&(data1['stimulus'] ==1)]
            data_High_No_0  = data1[(data1['coherency']=='High')&(data1['spatial']=='No')&(data1['stimulus'] ==0)]
            data_Low_Yes_1  = data1[(data1['coherency']=='Low')&(data1['spatial'] =='Yes')&(data1['stimulus']==1)]
            data_Low_Yes_0  = data1[(data1['coherency']=='Low')&(data1['spatial'] =='Yes')&(data1['stimulus']==0)]
            data_Low_No_1   = data1[(data1['coherency']=='Low')&(data1['spatial'] =='No')&(data1['stimulus'] ==1)]
            data_Low_No_0   = data1[(data1['coherency']=='Low')&(data1['spatial'] =='No')&(data1['stimulus'] ==0)]


            #mean rt of condition for data 
            data_rt_High_Yes_1_mean = np.mean(data_High_Yes_1['rt'])
            data_rt_High_Yes_0_mean = np.mean(data_High_Yes_0['rt']) 
            data_rt_High_No_1_mean  = np.mean(data_High_No_1['rt'])
            data_rt_High_No_0_mean  = np.mean(data_High_No_0['rt']) 
            data_rt_Low_Yes_1_mean  = np.mean(data_Low_Yes_1['rt']) 
            data_rt_Low_Yes_0_mean  = np.mean(data_Low_Yes_0['rt'])
            data_rt_Low_No_1_mean   = np.mean(data_Low_No_1['rt'])
            data_rt_Low_No_0_mean   = np.mean(data_Low_No_0['rt'])


            #mean acc of condition for data 
            data_acc_High_Yes_1_mean = float(np.sum(data_High_Yes_1['response']==1))/len(data_High_Yes_1)
            data_acc_High_Yes_0_mean = float(np.sum(data_High_Yes_0['response']==0))/len(data_High_Yes_0)
            data_acc_High_No_1_mean  = float(np.sum(data_High_No_1['response']==1))/len(data_High_No_1)
            data_acc_High_No_0_mean  = float(np.sum(data_High_No_0['response']==0))/len(data_High_No_0)
            data_acc_Low_Yes_1_mean  = float(np.sum(data_Low_Yes_1['response']==1))/len(data_Low_Yes_1)
            data_acc_Low_Yes_0_mean  = float(np.sum(data_Low_Yes_0['response']==0))/len(data_Low_Yes_0)
            data_acc_Low_No_1_mean   = float(np.sum(data_Low_No_1['response']==1))/len(data_Low_No_1)
            data_acc_Low_No_0_mean   = float(np.sum(data_Low_No_0['response']==0))/len(data_Low_No_0)
            
            #define new row for each subject to save in dataframe
            new_row = pd.DataFrame({'ACC' : [data_acc_High_Yes_1_mean*100, data_acc_High_Yes_0_mean*100, data_acc_High_No_1_mean*100,
                                             data_acc_High_No_0_mean*100, data_acc_Low_Yes_1_mean*100, data_acc_Low_Yes_0_mean*100,
                                            data_acc_Low_No_1_mean*100, data_acc_Low_No_0_mean*100],
                                     'RT': [data_rt_High_Yes_1_mean*1000, data_rt_High_Yes_0_mean*1000, data_rt_High_No_1_mean*1000,
                                             data_rt_High_No_0_mean*1000, data_rt_Low_Yes_1_mean*1000, data_rt_Low_Yes_0_mean*1000,
                                            data_rt_Low_No_1_mean*1000, data_rt_Low_No_0_mean*1000],
                                     'Condition' : ['High/Yes', 'High/Yes', 'High/No',
                                                    'High/No', 'Low/Yes', 'Low/Yes',
                                                    'Low/No', 'Low/No'],
                                     'Subject' : [select_subject,select_subject,select_subject,
                                                 select_subject,select_subject,select_subject,
                                                 select_subject,select_subject],
                                     'Stimulus' : ['Face','Car','Face',
                                                   'Car','Face','Car',
                                                   'Face','Car']})   
            performance_data = performance_data.append(new_row)

        
    performance_data['Subject'] = performance_data['Subject'].astype(int)

    #standard error of the mean rt
    SEM_RT = np.array([
    performance_data[(performance_data['Condition']=='High/Yes')&(performance_data['Stimulus']=='Car')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='High/Yes')&(performance_data['Stimulus']=='Face')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='High/No')&(performance_data['Stimulus']=='Car')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='High/No')&(performance_data['Stimulus']=='Face')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='Low/Yes')&(performance_data['Stimulus']=='Car')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='Low/Yes')&(performance_data['Stimulus']=='Face')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='Low/No')&(performance_data['Stimulus']=='Car')]['RT'].std(axis=0)/np.sqrt(14),
    performance_data[(performance_data['Condition']=='Low/No')&(performance_data['Stimulus']=='Face')]['RT'].std(axis=0)/np.sqrt(14)
    ])


    #plot bahavioral data (rt)
    g = sns.catplot(
        data=performance_data, kind="bar",
        x="Condition", y="RT", hue="Stimulus",
        ci=SEM_RT, palette="dark", alpha=.6, height=6, hue_order=['Face','Car']
    )

    g.despine(left=True)
    g.set_axis_labels("Coherecy/Prioritization", "Response Time (ms)")
    #g.legend.set_title("12")


    # statistical annotation
    x1, x2 = 2.7, 3.4   # stimulus
    y, h, col = 485 + 2, 2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col, fontsize=12)



    x1, x2 = 2, 3.1   # spatial
    y, h, col = 492 + 2, 2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col, fontsize=12)


    x1, x2 = 0.6, 2.7   # coherency
    y, h, col = 500 + 2, 2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "****", ha='center', va='bottom', color=col, fontsize=12)
    plt.show()


    #standard error of the mean acc
    SEM_ACC = np.array([
    performance_data[(performance_data['Condition']=='High/Yes')&(performance_data['Stimulus']=="Car")]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_data['Condition']=='High/Yes')&(performance_data['Stimulus']=='Face')]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_data['Condition']=='High/No')&(performance_data['Stimulus']=='Car')]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_data['Condition']=='High/No')&(performance_data['Stimulus']=='Face')]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_d ata['Condition']=='Low/Yes')&(performance_data['Stimulus']=='Car')]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_data['Condition']=='Low/Yes')&(performance_data['Stimulus']=='Face')]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_data['Condition']=='Low/No')&(performance_data['Stimulus']=='Car')]['ACC'].std(axis=0)/np.sqrt(1),
    performance_data[(performance_data['Condition']=='Low/No')&(performance_data['Stimulus']=='Face')]['ACC'].std(axis=0)/np.sqrt(1)
    ])



    # plot bahavioral data (accuracy)
    sns.set_style(style="ticks")
    g = sns.catplot(
        data=performance_data, kind="bar",
        x="Condition", y="ACC", hue="Stimulus",
        ci=SEM_ACC, palette="dark", alpha=.6, height=6, hue_order=['Face','Car']
    )
    g.despine(left=True)
    g.set_axis_labels("Coherecy/Prioritization", "Response Accuracy (%)")

     
    #g.legend.set_title("12")


    # statistical annotation
    x1, x2 = -.3, 0.3   # stimulus
    y, h, col = 99.5, 0.2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col, fontsize=12)



    x1, x2 = -.06, 1.05   # spatial
    y, h, col = 100.5, 0.2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col, fontsize=12)


    x1, x2 = 0.45, 2.6   # coherency
    y, h, col = 101.5, 0.2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "****", ha='center', va='bottom', color=col, fontsize=12)
    plt.show()


    #plor bahavioral data (RT distribution prioritized)
    plt.figure(figsize=(8,6))
    # high coherence and prioritized 
    sns.distplot(data[(data['coherency']=='High')&
                      (data['spatial']=='Yes')&
                      (data['stimulus']==1)]['rt'].to_numpy()*1000, label='HF')
    sns.distplot(data[(data['coherency']=='High')&
                      (data['spatial']=='Yes')&
                      (data['stimulus']==0)]['rt'].to_numpy()*1000, label='HC')
    # Low coherence and prioritized 
    sns.distplot(data[(data['coherency']=='Low')&
                      (data['spatial']=='Yes')&
                      (data['stimulus']==1)]['rt'].to_numpy()*1000, label='LF')
    sns.distplot(data[(data['coherency']=='Low')&
                      (data['spatial']=='Yes')&
                      (data['stimulus']==0)]['rt'].to_numpy()*1000, label='LC')
    sns.despine(offset=10, trim=True);



    #plor bahavioral data (RT distribution prioritized)
    plt.figure(figsize=(8,6))
    # high coherence and non-prioritized 
    sns.distplot(data[(data['coherency']=='High')&
                      (data['spatial']=='No')&
                      (data['stimulus']==1)]['rt'].to_numpy()*1000, label='HF')
    sns.distplot(data[(data['coherency']=='High')&
                      (data['spatial']=='No')&
                      (data['stimulus']==0)]['rt'].to_numpy()*1000, label='HC')
    # Low coherence and Non-prioritized 
    sns.distplot(data[(data['coherency']=='Low')&
                      (data['spatial']=='No')&
                      (data['stimulus']==1)]['rt'].to_numpy()*1000, label='LF')


    sns.distplot(data[(data['coherency']=='Low')&
                      (data['spatial']=='No')&
                      (data['stimulus']==0)]['rt'].to_numpy()*1000, label='LC')
    sns.despine(offset=10, trim=True);
# -------------------------------------------------------------------------    


# -------------------------------------------------------------------------    

#3*3 repeaded measure anova
def ANOVA_PDM():
    data = pd.read_csv('PDM_data\\filtered_anlz_group\\outside\\group_outside.csv')
    #elimination subject number 7 and 11 relating to eeg detect data
    #data = data[(data['subj_idx']!=7)&(data['subj_idx']!=11)]

    data['coherency'] = data['condition']
    data['spatial'] = data['condition']
        
    #just spatial condition
    data['spatial']=data['spatial'].replace(1,'Yes')
    data['spatial']=data['spatial'].replace(2,'No')
    data['spatial']=data['spatial'].replace(3,"Yes")
    data['spatial']=data['spatial'].replace(4,'No')
        
    #just coherency condition
    data['coherency']=data['coherency'].replace(1,'High')
    data['coherency']=data['coherency'].replace(2,'High')
    data['coherency']=data['coherency'].replace(3,"Low")
    data['coherency']=data['coherency'].replace(4,'Low')

    #face 1
    #car 0
    #rersponse
    data['response']=data['response'].replace(1,0)
    data['response']=data['response'].replace(2,1)

    #stimulus
    data['stimulus']=data['stimulus'].replace(1,0)
    data['stimulus']=data['stimulus'].replace(2,1)

    data_filter = data[(data['subj_idx']!=7)&(data['subj_idx']!=11)]
    aovrm2way = AnovaRM(data_filter, 'rt', 'subj_idx', within=['coherency','spatial', 'stimulus'], aggregate_func='mean')
    res2way = aovrm2way.fit()
    return res2way
# -------------------------------------------------------------------------    

        
