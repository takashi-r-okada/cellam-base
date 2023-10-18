# coding: utf-8

# 初期化も含めて行う。けど別途任意の初期状態を入れてもいいようにする。
# このスクリプトの前提: 2 値、3x3 隣接セルで固定。taxon 色塗りはこの中では行わない

import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import random
from PIL import Image, ImageDraw
import scipy.stats as stats
import yaml
import sys
import mlflow
from pathlib import Path
import shutil

with open(sys.argv[1], 'r', encoding='utf-8-sig') as yml:
    cfg = yaml.safe_load(yml)


L = cfg['L']
T = cfg['T']
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])


result_temp_dp = Path(r"./result_temp_dir/")
if result_temp_dp.exists():
    print('結果フォルダを初期化．')
    shutil.rmtree(result_temp_dp)
result_temp_dp.mkdir(parents=True, exist_ok=False)
mlflow.set_tracking_uri(cfg['mlflow-tracking-uri'])
mlflow.set_experiment(cfg['mlflow-exp-name'])
with mlflow.start_run(run_name=cfg['mlflow-run-name']):
    
    mlflow.log_params(cfg)

    # --------------------------------------------------
    # 初期世界生成
    # --------------------------------------------------

    world = np.zeros((L, L), dtype=np.float32)

    for y in range(world.shape[0]):
        for x in range(world.shape[1]):
            if random.random() < cfg['initial-alive-prob']:
                world[y, x] = 1

    world_prev = copy.deepcopy(world)

    for y in range(1, world.shape[0]-1):
        for x in range(1, world.shape[1]-1):
            if np.sum(world_prev[y-1: y+2, x-1: x+2]) >= 2:
                world[y, x] = 1

    del world_prev

    world[:cfg['yrange'][0], :] = 0
    world[cfg['yrange'][1]:, :] = 0
    world[:, :cfg['xrange'][0]] = 0
    world[:, cfg['xrange'][1]:] = 0

    cv2.imwrite(str(result_temp_dp / 'world-init.png'), world * 255.)
    mlflow.log_artifact(result_temp_dp / 'world-init.png')



    
    # mlflow.log_param("", n_estimators)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_metric("mse", mse)
    # mlflow.log_metric("R2", R2)

    # mlflow.sklearn.log_model(model, "model")





    def mapFunc(state: np.ndarray):
        '''
        [引数]
        - state: [3, 3] array
        '''

        if state[1, 1] == 0:
            if np.sum(state) == 3:
                return 1
            # elif np.sum(state) == 2:
            #     if state[0,0] + state[2,2] == 2:
            #         return 1
            #     elif state[0,1] + state[2,1] == 2:
            #         return 1
            #     elif state[0,2] + state[2,0] == 2:
            #         return 1
            #     elif state[1,0] + state[1,2] == 2:
            #         return 1
            #     else:
            #         return 0
            else:
                return 0
        else:
            if np.sum(state) in [2+1, 3+1]:
                return 1
            else:
                return 0



    def wUpdate(world):
        world_prev = copy.deepcopy(world)
        for y in range(1, world.shape[0]-1):
            for x in range(1, world.shape[1] - 1):
                world[y, x] = mapFunc(world_prev[y-1: y+2, x-1: x+2])
        del world_prev


    # --------------------------------------------------
    # 時間発展
    # --------------------------------------------------
    result = np.zeros((T, L, L), dtype=np.float32)
    result[0] = world
    for t in range(T-1):
        print("\r[time dev.] t = %d" % t, end="")
        wUpdate(world)
        result[t+1]=world

    # --------------------------------------------------
    # 新逗子
    # --------------------------------------------------

    def newMod(a, b):
        # a % b の代わりを作る
        # a は配列
        # b は自然数

        ret = a % b
        ret[(a != 0) & (ret == 0)] = b//2
        return ret
    


    P = 32
    Q = 0.005
    def getColoredInitMap(mat):
        matU8 = np.clip(mat * 255., 0, 255).astype(np.uint8)
        retval, labels, _, _ = cv2.connectedComponentsWithStats(matU8, connectivity=8)
        del matU8

        # newLabels = copy.deepcopy(labels)
        # newLabels[newLabels == 0] = None
        # labels = labels % P
        labels = newMod(labels, P)
        
        matC_hsv = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)
        matC_hsv[:, :, 0] = labels / np.max(labels) * 179.
        matC_hsv[:, :, 1] = 255
        matC_hsv[:, :, 2] = 255
        
        matC_hsv[labels == 0] = [0, 0, 0]
        
        matC = cv2.cvtColor(matC_hsv,cv2.COLOR_HSV2BGR)
        return matC, labels

    # cMat, cLabels = getColoredInitMap(result[3, ])
    # plt.imshow(cMat)




    def getColoredProceedingMap(mat, previousLabels):
        matU8 = np.clip(mat * 255., 0, 255).astype(np.uint8)
        matU8[:2, :2] = 0
        retval, labels, _, _ = cv2.connectedComponentsWithStats(matU8, connectivity=8)
        del matU8

        pMats = np.zeros((P, mat.shape[0], mat.shape[1]), dtype=np.uint8)
        for _p in range(0, P):
            pMats[_p] = (previousLabels == _p)

        matC_hsv = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)
        matC_h = np.zeros((mat.shape[0], mat.shape[1]), dtype=np.uint8)

        pMats_flat = pMats.reshape(P, -1)
        labels_flat = labels.reshape(-1)

        for b in range(1, retval):
            
            _adPMats = pMats_flat * (labels_flat == b)
            _adPMats = np.sum(_adPMats, axis=1)
            
            if np.max(_adPMats[1:]) == 0:
                newLabel = random.randint(1, 15)
            else:
                newLabel = np.argmax(_adPMats[1:]) + 1

            if random.random() < Q:
                newLabel = random.randint(1, 15)

            matC_h[labels == b] = newLabel / 15. * 179.
            labels[labels == b] = newLabel
            del newLabel

        matC_hsv[:, :, 0] = matC_h
        matC_hsv[:, :, 1] = 255
        matC_hsv[:, :, 2] = 255

        matC_hsv[labels == 0] = [0, 0, 0]
        matC = cv2.cvtColor(matC_hsv,cv2.COLOR_HSV2BGR)

        return matC, labels

    # cMat, cLabels = getColoredProceedingMap(result[4], cMat, cLabels)
    # plt.imshow(cMat)
    # plt.show()
    # cMat, cLabels = getColoredProceedingMap(result[5], cMat, cLabels)
    # plt.imshow(cMat)
    # plt.show()





    # DISPLAY_MIN_MAX = [0.25, 0.75]
    DISPLAY_MIN_MAX = [0.125, 0.875]
    DISPLAY_SCALE_FACTOR = 2.
    DISPLAY_STOP_EPOCH = T
    DISPLAY_START_EPOCH = 1

    displayRange = [int(round(L * DISPLAY_MIN_MAX[0])), int(round(L * DISPLAY_MIN_MAX[1]))]
    displaySize = int(round((displayRange[1] -displayRange[0]) * DISPLAY_SCALE_FACTOR))


    def cv2pil(image):
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image

    cMats = []
    for fid, mat in enumerate(result):
        if fid > DISPLAY_STOP_EPOCH:
            break
        print("\r%d" % fid, end="")
        if fid < DISPLAY_START_EPOCH:
            continue

        _mat = mat[displayRange[0]: displayRange[1] ,displayRange[0]: displayRange[1]]
        
        if fid == DISPLAY_START_EPOCH:
            cMat, cLabels = getColoredInitMap(_mat)
        else:
            cMat, cLabels = getColoredProceedingMap(_mat, cMat, cLabels)

        cMats.append(cMat)




    imgs = []
    for fid, mat in enumerate(cMats):
        if fid > DISPLAY_STOP_EPOCH:
            break
        if fid < DISPLAY_START_EPOCH:
            continue


        _top = L//3 - displayRange[0]
        _bottom = L//3 - displayRange[0] + L//3
        _left = L//3 - displayRange[0]
        _right = L//3 - displayRange[0] + L//3
        
        _mat = cv2.rectangle(copy.deepcopy(mat), (_left, _top), (_right, _bottom), (0, 191, 0), thickness=1)
        # _mat = cv2.rectangle(_mat, (_left, _top), (_right, _bottom), (0, 191, 0), thickness=1)
        _mat = cv2.putText(_mat, "t=%d" % (fid + DISPLAY_START_EPOCH), (3, 11), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)

        

        
        imgs.append(cv2pil(
            cv2.resize(
                _mat,
                (displaySize, displaySize),
                interpolation=cv2.INTER_NEAREST
                # interpolation=cv2.INTER_CUBIC
            )
        ))

    plt.imshow(_mat)





    FPS = 15

    imgs[0].save(
        result_temp_dp / "colored-result-anime.gif",
        # 'lifeGame4t.gif',
        save_all=True,
        # append_images=imgs[1:500],
        append_images=imgs[1:],
        optimize=False,
        duration=1000//FPS,
        loop=0
    )

    "done"