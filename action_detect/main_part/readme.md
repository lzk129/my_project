项目运行方式如下：
1.python main_part/lab.py
在lab.py中，将第一个路径改为自己视频所在的路径，第二个是视频帧的保存路径。
2.python main_part/first.py
在first.py中，输入为视频帧，也就是上一步的输出，将path改为对应地视频帧文件路径。
输出有两个，一个是path2，一个是filepath，第一个是骨架图，第二个是骨架坐标信息，修改为自己的文件路径。
3.python main_part/smt.py
输入是上一步的输出，将input_path改为骨架坐标信息的文件路径，输出有两个，改为自己的文件路径.
4.python main_part/tra.py
输入时上一步的输出，也就是两个csv文件，将路径修改为自己的文件路径。
记录准确率。