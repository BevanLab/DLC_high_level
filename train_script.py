import deeplabcut

config = 'D:\\DLC-training\\paw_40_labels-Ameet-2021-04-02\\config.yaml'
shuffle = 1
trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(config, shuffle=1 )
cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)

maxiters = 200000
cfg_dlc['scale_jitter_lo']= 0.5
cfg_dlc['scale_jitter_up']= 1.5

cfg_dlc['augmentationprobability']=.5
cfg_dlc['batch_size']=1 #pick that as large as your GPU can handle it
cfg_dlc['elastic_transform']=True
cfg_dlc['rotation']=180
cfg_dlc['covering']=True
cfg_dlc['motion_blur'] = True
cfg_dlc['optimizer'] ="adam"
cfg_dlc['dataset_type']='imgaug'
cfg_dlc['multi_step']=[[1e-4, 7500], [5.0e-5, 12000], [1e-5, 50000], [1e-5, maxiters]]

deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

print("TRAIN NETWORK", shuffle)
deeplabcut.train_network(config, shuffle=shuffle,saveiters=1000,displayiters=500,maxiters = maxiters,max_snapshots_to_keep=11, gputouse = 1)