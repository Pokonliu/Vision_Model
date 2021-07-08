import sys
import math
import numpy as np
import cv2
# from matplotlib import pyplot as plt
# from skimage import morphology

btn_down = False
needle_width = 32
needle_height = 200

def getROI(im, ROI=None):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()  # python的字典的复制，返回一个字典的浅复制

    if ROI is None:
        #  Set the callback function for any mouse event
        cv2.imshow("Image", im)
        cv2.setMouseCallback("Image", getROIMouseHandler, data)
        cv2.waitKey(0)

        # Convert array to np.array in shape n,2,2
        ROI = data['ROI']

        data['im'] = data['im'][ROI[0][1]:ROI[1][1]+1, ROI[0][0]:ROI[1][0]+1]
        cv2.destroyWindow("Image")
    else:
        data['im'] = data['im'][ROI[0][1]:ROI[1][1]+1, ROI[0][0]:ROI[1][0]+1]

    return data['im'], ROI


def getROIMouseHandler(event, x, y, flags, data):
    img2 = data['im'].copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        data['point1'] = (x, y)
        cv2.circle(img2, data['point1'], 1, (100, 200, 50), 1)
        cv2.imshow('Image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, data['point1'], (x, y), (200, 100, 50), 2)
        cv2.imshow('Image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        data['point2'] = (x, y)

        min_x = max(min(data['point1'][0], data['point2'][0]),0)
        min_y = max(min(data['point1'][1], data['point2'][1]),0)
        max_x = min(max(data['point1'][0], data['point2'][0]),img.shape[1])
        max_y = min(max(data['point1'][1], data['point2'][1]),img.shape[0])
        # img.shape = (H,W,C)
        data['ROI'] = [[min_x, min_y], [max_x, max_y]]

        cv2.rectangle(img2, (min_x, min_y), (max_x, max_y), (100, 100, 200), 2)
        cv2.imshow('Image', img2)



def getLines(im, rq_num_needles=False):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()  # python的字典的复制，返回一个字典的浅复制
    data['points'] = []

    #  Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", getLinesMouseHandler, data)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.float32(data['points'])
    # 输入针格数
    if rq_num_needles:
        # num_needles=10
        print("请勿关闭图片，划线区间有多少个针格？")
        num_needles = sys.stdin.readline()
        num_needles = int(num_needles)
        cv2.destroyWindow("Image")
        return points, num_needles
    else:
        cv2.destroyWindow("Image")
        return points


def getLinesMouseHandler(event, x, y, flags, data):
    global btn_down
    if event == cv2.EVENT_LBUTTONUP and btn_down:
        #  if you release the button, finish the line
        btn_down = False
        data['points'][-1].append((x, y))  #  append the second point
        cv2.circle(data['im'], (x, y), 3, (100, 100, 200),5,16)
        cv2.line(data['im'], data['points'][-1][0], data['points'][-1][1], (100, 100, 200), 1, cv2.LINE_AA)
        cv2.imshow("Image", data['im'])

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #this is just for a line visualization
        image = data['im'].copy()
        cv2.line(image, data['points'][-1][0], (x, y), (200,100,100), 1, cv2.LINE_AA)
        cv2.imshow("Image", image)

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['points']) < 3:
        btn_down = True
        data['points'].append([(x, y)])  #  append the point
        cv2.circle(data['im'], (x, y), 3, (100, 100, 200), 5, 16)
        cv2.imshow("Image", data['im'])


def caculateAverageAngle(lines):
    num_line = len(lines)
    theta_list = []
    for line in lines:
        star_point, end_point = line # point = [x,y]
        x_delta = star_point[0] - end_point[0]
        y_delta = star_point[1] - end_point[1]
        if x_delta == 0:
            if y_delta == 0:
                raise Exception("The input line is a dot.")
            else:
                angel = math.pi/2
        tan_theta = y_delta / x_delta
        theta = math.atan(tan_theta)
        theta_list.append(theta)
    
    theta = np.average(theta_list)
    return theta


def rotateImg(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    W,H = img.shape[1::-1]
    
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    rotate_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
    
    # 计算旋转后输出图形的尺寸
    rotated_width = math.ceil(H * math.fabs(rotate_matrix[0,1])
                            + W * math.fabs(rotate_matrix[1,1]))
    rotated_height = math.ceil(W * math.fabs(rotate_matrix[0,1])
                            + H * math.fabs(rotate_matrix[1,1]))
    
    # 防止切边，对平移矩阵B进行修改
    rotate_matrix[0,2] += (rotated_width - W) / 2
    rotate_matrix[1,2] += (rotated_height - H) / 2
    
    # 应用旋转
    img_rotated = cv2.warpAffine(img, rotate_matrix, (rotated_width, rotated_height))
    
    return img_rotated


def cutImg():
    pass


# 方案三：手工设计检测方案。需要假设待检测直线是水平或垂直的。(已通过，耗时约0.6ms，需通过实验验证假设是否合理。)
def hand_designed_detector(detect_window_bin):
    H_detect_window,W_detect_window = detect_window_bin.shape # (96, 38)， 这是经过反转的二值图像，标志线应为白色，原来的黑背景也变成白色
    y_sum = (detect_window_bin>0).sum(axis=1) # (96,)
    print(y_sum)
    length_thld = 4
    y_line = y_sum>(W_detect_window-length_thld) # 如果该行水平方向白色像素够多，就为True，反之为False。
    print(y_line)
    assert y_line[0] == True and y_line[-1] == True # 由于背景存在，首末两端应该都是白色

    # 去除首末两端的白色区段
    star_point = 0
    while y_line[star_point] == True:
        y_line[star_point] = False
        star_point += 1
    end_point = H_detect_window-1
    while y_line[end_point] == True:
        y_line[end_point] = False
        end_point -= 1
    print(star_point, end_point)
    # 确认只剩下两处白色区段，并计算平均坐标
    coor_list = []
    y_coor = []
    for i in range(star_point,end_point+1):
        if y_line[i] == False and coor_list != []:
            y_coor.append(int(np.average(coor_list)))
            coor_list = []
        elif y_line[i] == True:
            coor_list.append(i)
    assert len(y_coor)==2

    # 在选定区间找竖线
    detect_window_bin_narrow = detect_window_bin[y_coor[0]:y_coor[1],:]
    x_sum = (detect_window_bin_narrow>0).sum(axis=0)
    length_thld = 2
    x_line = x_sum>(detect_window_bin_narrow.shape[0]-length_thld) # 如果该行垂直方向白色像素够多，就为True，反之为False。

    # 确认白色区段小于两处，并计算平均坐标
    coor_list = []
    x_coor = []
    for i in range(len(x_sum)):
        if x_line[i] == False and coor_list != []:
            x_coor.append(int(np.average(coor_list)))
            coor_list = []
        elif x_line[i] == True:
            coor_list.append(i)
    assert len(x_coor)<=2

    if len(x_coor)==2: # 如果有两条竖线就选个离窗口中心近的，靠边的不准
        if np.abs(W_detect_window/2-x_coor[0]) < np.abs(W_detect_window/2-x_coor[1]):
            x_coor = x_coor[0]
        else:
            x_coor = x_coor[1]
    else:
        x_coor = x_coor[0]
    return y_coor, x_coor







if __name__ == "__main__":
    img = cv2.imread('./Image_20210629160819614.bmp')#[:100,:200,:]

    # ROI(Region of Interest)初选
    print('ROI(Region of Interest)初选')
    image_ROI, ROI = getROI(img)

    # 旋转基准线
    print('画旋转基准线')
    lines_rotate = getLines(image_ROI)
    print("Lines coordinates:\n", lines_rotate)
    angle = caculateAverageAngle(lines=lines_rotate)
    angle = angle/math.pi*180 # change unit from rad to degree
    print("Angel (degree): ", angle)
    rotatedImg = rotateImg(image_ROI, angle=angle)

    # 放缩辅助线
    print('画放缩辅助线')
    lines_scale, num_needles = getLines(rotatedImg, rq_num_needles=True)
    print("Lines coordinates:\n", lines_scale)
    line_range = abs(lines_scale[-1][0][0] - lines_scale[-1][1][0])
    scale = num_needles * needle_width / line_range
    print("Scale: ", scale)
    rotatedResizedImg = cv2.resize(rotatedImg, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


    # 画水平中线
    print('画水平中线（上）')
    horizon_line1 = getLines(rotatedResizedImg)[-1]
    print('画水平中线（下）')
    horizon_line2 = getLines(rotatedResizedImg)[-1]
    print("Lines coordinates:\n", horizon_line1, "\n", horizon_line2)
    mid_point = int((horizon_line1[0][1] + horizon_line1[1][1] + 
                     horizon_line2[0][1] + horizon_line2[1][1]) / 4)
    assert mid_point - needle_height >= 0 and mid_point + needle_height < rotatedResizedImg.shape[0]
    midCroppedImg = rotatedResizedImg[mid_point-needle_height:mid_point+needle_height,:,:]

    # 对于同一台机器的同一个视点，初选ROI, angle, scale, horizon_line 都可以重复利用。
    # 仅以上参数的计算需人工画辅助线，下面精选ROI可自动化。
    
    # 截取定位图案检测窗口
    H_midCropped,W_midCropped,C_midCropped = midCroppedImg.shape
    new_mid_point = needle_height
    pattern_from_mid = 120 # 预定义的位置常量
    detect_window = midCroppedImg[needle_height+pattern_from_mid:needle_height+pattern_from_mid+3*needle_width,
                    W_midCropped//2:W_midCropped//2+int(1.2*needle_width),:] # 3:1.2的窗口，以针格宽度为单位 # (96, 38)
    # 开始检测标志直线
    bin_thld = 100 # 二值化阈值
    detect_window_gray = detect_window[:,:,1]
    detect_window_bin = (detect_window_gray<100).astype('uint8')*255

    # 方案一：提取边缘+霍夫直线检测（未通过，耗时约0.7ms,i7-4930K）。难点：识别出较多直线（尺子边缘+标志线较粗形成双边缘），需要制定方案来选择合适的直线。
    # edges = cv2.Canny(detect_window,100,200)
    # cdstP1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, None, 25, 2)
    # # Draw the lines
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP1, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    # cv2.imshow("detect_window", detect_window)
    # cv2.imshow("edges", edges)
    # cv2.imshow("cdstP1", cdstP1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # 方案二：骨架提取+直线检测 (未通过，耗时约5ms，开销太大，可能需要多线程，或者先把两边背景区域填充，也许能降低计算量)。难点：有的骨架扭曲导致直线检测失败,可以尝试滤波、膨胀腐蚀等预处理，再做骨架提取。
    # skeleton0 = morphology.skeletonize(detect_window_bin>0)
    # skeleton = (skeleton0[:,:]>0).astype('uint8')*255
    # cdstP2 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    # linesP = cv2.HoughLinesP(skeleton, 1, np.pi / 180, 25, None, 25, 2)
    # # Draw the lines
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    # cv2.imshow("detect_window_bin", detect_window_bin)
    # cv2.imshow("skeleton", skeleton)
    # cv2.imshow("cdstP2", cdstP2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 方案三：手工设计检测方案。需要假设待检测直线是水平或垂直的。(已通过，耗时约0.6ms，需通过实验验证假设是否合理。)
    y_coor, x_coor = hand_designed_detector(detect_window_bin)  # 输入检测窗口图片，输出标志线坐标
    ## 计算标志线在预处理图像的原始坐标
    y_coor = [(needle_height+pattern_from_mid) + y for y in y_coor] # (needle_height+pattern_from_mid)是检测窗口纵坐标起点
    x_coor = x_coor + W_midCropped//2 # W_midCropped//2是检测窗口横坐标起点

    midCroppedImgCopy = midCroppedImg.copy()
    cv2.line(midCroppedImgCopy, (x_coor, y_coor[0]), (x_coor, y_coor[1]), (100,100,255), 1, cv2.LINE_AA)
    cv2.imshow("midCroppedImgCopy", midCroppedImgCopy) # 画出标志线
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 读取相邻3个二维码，用十六进制表示
    midCroppedImg_bin = midCroppedImg[:,:,1]>100 # 二值化
    code_x_list = [x_coor-needle_width, x_coor, x_coor+needle_width]
    point_y_offset =[9,21]
    point_x_offset =[10,22]
    code_list = []
    y = y_coor[0]
    for x in code_x_list:
        point0 = midCroppedImg_bin[y + point_y_offset[0], x + point_x_offset[0]]
        midCroppedImgCopy[y + point_y_offset[0], x + point_x_offset[0],:] = (100,255,100)
        point1 = midCroppedImg_bin[y + point_y_offset[0], x + point_x_offset[1]]
        midCroppedImgCopy[y + point_y_offset[0], x + point_x_offset[1],:] = (100,255,100)
        point2 = midCroppedImg_bin[y + point_y_offset[1], x + point_x_offset[0]]
        midCroppedImgCopy[y + point_y_offset[1], x + point_x_offset[0],:] = (100,255,100)
        point3 = midCroppedImg_bin[y + point_y_offset[1], x + point_x_offset[1]]
        midCroppedImgCopy[y + point_y_offset[1], x + point_x_offset[1],:] = (100,255,100)
        code_list.append(f'{point0:d}{point1:d}{point2:d}{point3:d}')
    cv2.imshow("midCroppedImgCopy", midCroppedImgCopy) # 画出读点位置
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    code_list = [f'{int(i,2):x}' for i in code_list] # 进制转换
    code = code_list[0] + code_list[1] + code_list[2] # 3位十六进制组成的code，通过查字典得到当前位置
    db_dict = np.load('db_dict.npy', allow_pickle=True).flat[0]
    print(db_dict)
    position = db_dict[code]






