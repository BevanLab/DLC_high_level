import deeplabcut

config = 'D:\\DLC-training\\paw_40_labels-Ameet-2021-04-02\\config.yaml'


videofile_path = ["E:\\MitoPark\\Open Field\\DLC_MP_test\\"]
VideoType = 'mp4'
gpu = 1
deeplabcut.analyze_videos(config ,videofile_path, videotype=VideoType, save_as_csv=True, gputouse = gpu)
deeplabcut.plot_trajectories(config ,videofile_path, videotype=VideoType)
deeplabcut.create_labeled_video(config ,videofile_path, videotype=VideoType, draw_skeleton= False , save_frames=True)