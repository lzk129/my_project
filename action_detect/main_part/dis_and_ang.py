from __future__ import division
import os

from matplotlib.pyplot import bone
import cv2
import math

class get_math_information(object):
    def get_distance(self,A,B):
        if A is None or B is None:#warning: A or B is a tuple instead of a number
            return 0
        else:
            return math.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)
    
    def get_angle(self,A,B,C):
        if A is None or B is None or C is None:
            return 0
        else:
            a=self.get_distance(B,C)
            b=self.get_distance(A,C)
            c=self.get_distance(A,B)
            if a*c!=0:
                return math.degrees((a**2/+b**2-c**2)/(2*a*c))
            return 0

    def get_skeleton_disiance(self,bone_points):
        distance=[]
        if len(bone_points)==18:
            distance.append(self.get_distance(bone_points[0],bone_points[1]))#头和脖子距离
            distance.append(self.get_distance(bone_points[5],bone_points[7]))#手肩
            distance.append(self.get_distance(bone_points[2],bone_points[4]))
            distance.append(self.get_distance(bone_points[4],bone_points[8]))#右手右腰
            distance.append(self.get_distance(bone_points[7],bone_points[11]))#左手左腰
            distance.append(self.get_distance(bone_points[8],bone_points[10]))#右腿右腰
            distance.append(self.get_distance(bone_points[11],bone_points[13]))#左腿左腰
            distance.append(self.get_distance(bone_points[0],bone_points[4]))#头和右手
            distance.append(self.get_distance(bone_points[0],bone_points[7]))#头和右手
            distance.append(self.get_distance(bone_points[1],bone_points[4]))#右手脖子
            distance.append(self.get_distance(bone_points[1],bone_points[7]))#左手脖子
            distance.append(self.get_distance(bone_points[1],bone_points[11]))#脖子右腰
            distance.append(self.get_distance(bone_points[1],bone_points[8]))#脖子左腰
            distance.append(self.get_distance(bone_points[16],bone_points[17]))#双耳
            distance.append(self.get_distance(bone_points[4],bone_points[16]))#右手右耳
            distance.append(self.get_distance(bone_points[7],bone_points[2]))#左手右肩
            distance.append(self.get_distance(bone_points[7],bone_points[3]))#左手右臂
        else:
            distance=None

        return distance
    
    def get_skeleton_angle(self,bone_points):
        angle=[]
        angle.append(self.get_angle(bone_points[3],bone_points[2],bone_points[1]))#右臂和肩
        angle.append(self.get_angle(bone_points[4],bone_points[3],bone_points[2]))#右手右臂
        angle.append(self.get_angle(bone_points[6],bone_points[5],bone_points[1]))#左臂和肩
        angle.append(self.get_angle(bone_points[7],bone_points[6],bone_points[5]))#左手左臂
        angle.append(self.get_angle(bone_points[4],bone_points[0],bone_points[7]))#头手头
        if bone_points[8] is None or bone_points[11] is None:
            angle.append(0)
        else:
            temp=((bone_points[8][0]+bone_points[11][0])/2,(bone_points[8][1]+bone_points[11][1])/2)#算出两腰之间的距离
            angle.append(self.get_angle(bone_points[4],temp,bone_points[7]))#手腰手
        angle.append(self.get_angle(bone_points[4],bone_points[1],bone_points[8]))#右手脖腰
        angle.append(self.get_angle(bone_points[7],bone_points[1],bone_points[11]))#左手脖腰

        return angle


        


