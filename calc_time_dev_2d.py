# coding: utf-8

# 初期化も含めて行う。けど別途任意の初期状態を入れてもいいようにする。
# このスクリプトの前提: 2 値、3x3 隣接セルで固定。

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

import utility_gif2mp4

from modules import lib_math, lib_pil
from time_dev_rules.lib_time_dev_rules_2d import TIME_DEV_RULES_2D


with open(sys.argv[1], 'r', encoding='utf-8-sig') as yml:
    cfg = yaml.safe_load(yml)

L = cfg['L']
T = cfg['T']
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])

# --------------------------------------------------
# 時間発展規則の用意
# --------------------------------------------------

mapRule = TIME_DEV_RULES_2D[cfg['time-dev-rule']]

# --------------------------------------------------
# 時間発展関数の用意
# --------------------------------------------------

def wUpdate(world):
    world_prev = copy.deepcopy(world)
    for y in range(1, world.shape[0]-1):
        for x in range(1, world.shape[1] - 1):
            world[y, x] = mapRule(world_prev[y-1: y+2, x-1: x+2])
    del world_prev

# --------------------------------------------------
# 出力関係・mlflow の設定
# --------------------------------------------------

result_temp_dp = Path(r"./result_temp_dir/")
if result_temp_dp.exists():
    print('結果フォルダを初期化．')
    shutil.rmtree(result_temp_dp)
result_temp_dp.mkdir(parents=True, exist_ok=False)
mlflow.set_tracking_uri(cfg['mlflow-tracking-uri'])
mlflow.set_experiment(cfg['mlflow-exp-name'])
with mlflow.start_run(run_name=cfg['mlflow-run-name']):
    
    mlflow.log_params(cfg)
    mlflow.log_artifact(sys.argv[1])

    # --------------------------------------------------
    # 初期世界生成
    # --------------------------------------------------

    world = np.zeros((L, L), dtype=np.float32)

    if 'custom-initial-array' in cfg:

        customInitalArray = np.array(cfg['custom-initial-array'])
        world[
            world.shape[0] // 2 - customInitalArray.shape[0] // 2: world.shape[0] // 2 - customInitalArray.shape[0] // 2 + customInitalArray.shape[0],
            world.shape[1] // 2 - customInitalArray.shape[1] // 2: world.shape[1] // 2 - customInitalArray.shape[1] // 2 + customInitalArray.shape[1],
        ] = customInitalArray

    else:

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

    # --------------------------------------------------
    # 時間発展
    # --------------------------------------------------

    result = np.zeros((T+1, L, L), dtype=np.float32)
    result[0] = world
    for t in range(T):
        print("\r[time dev.] t: %d -> %d ." % (t, t+1), end="")
        wUpdate(world)
        result[t+1]=world
    print()

    # --------------------------------------------------
    # 新逗子
    # --------------------------------------------------


    if cfg['display-method']['type'] == "trace-taxon-cmap":

        spaceBoundary = cfg['display-method']['space-boundary']
        zoomFactor = cfg['display-method']['zoom-factor']
        startEpoch = cfg['display-method']['start-epoch']
        stopEpoch = cfg['display-method']['stop-epoch']
        nColor = cfg['display-method']['nColor']
        fps = cfg['display-method']['fps']
        q = cfg['display-method']['q']
        drawGreenRect = cfg['display-method']['draw-green-rect']


        displayRange = [int(round(L * spaceBoundary[0])), int(round(L * spaceBoundary[1]))]
        displaySize = int(round((displayRange[1] -displayRange[0]) * zoomFactor))
        del spaceBoundary
        # del displayRange
        del zoomFactor

        # --------------------------------------------------
        # 生アニメ生成
        # -------------------------------------------------

        imgs = []
        for fid, mat in enumerate(result):
            if fid > stopEpoch:
                break

            _mat = mat[displayRange[0]: displayRange[1] ,displayRange[0]: displayRange[1]]
            _mat = np.clip(_mat * 255., 0, 255).astype(np.uint8)
            
            imgs.append(lib_pil.cv2pil(
                cv2.resize(
                    _mat,
                    (displaySize, displaySize),
                    interpolation=cv2.INTER_NEAREST
                    # interpolation=cv2.INTER_CUBIC
                )
            ))

        imgs[0].save(
            result_temp_dp / "raw-result-anime.gif",
            save_all=True,
            append_images=imgs[1:],
            optimize=False,
            duration=1000//fps,
            loop=0
        )
        mlflow.log_artifact(result_temp_dp / "raw-result-anime.gif")
        del imgs
        del _mat


        # --------------------------------------------------
        # taxon 着色アニメ生成
        # --------------------------------------------------
    
        def getColoredInitMap(mat):
            matU8 = np.clip(mat * 255., 0, 255).astype(np.uint8)
            retval, labels, _, _ = cv2.connectedComponentsWithStats(matU8, connectivity=8)
            del matU8

            labels = lib_math.newMod(labels, nColor)
            
            matC_hsv = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)
            matC_hsv[:, :, 0] = labels / np.max(labels) * 179.
            matC_hsv[:, :, 1] = 255
            matC_hsv[:, :, 2] = 255
            
            matC_hsv[labels == 0] = [0, 0, 0]
            
            matC = cv2.cvtColor(matC_hsv,cv2.COLOR_HSV2BGR)
            return matC, labels


        def getColoredProceedingMap(mat, previousLabels):
            matU8 = np.clip(mat * 255., 0, 255).astype(np.uint8)
            matU8[:2, :2] = 0
            retval, labels, _, _ = cv2.connectedComponentsWithStats(matU8, connectivity=8)
            del matU8

            pMats = np.zeros((nColor, mat.shape[0], mat.shape[1]), dtype=np.uint8)
            for _p in range(0, nColor):
                pMats[_p] = (previousLabels == _p)

            matC_hsv = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)
            matC_h = np.zeros((mat.shape[0], mat.shape[1]), dtype=np.uint8)

            pMats_flat = pMats.reshape(nColor, -1)
            labels_flat = labels.reshape(-1)

            for b in range(1, retval):
                
                _adPMats = pMats_flat * (labels_flat == b)
                _adPMats = np.sum(_adPMats, axis=1)
                
                if np.max(_adPMats[1:]) == 0:
                    newLabel = random.randint(1, 15)
                else:
                    newLabel = np.argmax(_adPMats[1:]) + 1

                if random.random() < q:
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



        cMats = []
        for fid, mat in enumerate(result):
            if fid > stopEpoch:
                break
            print("\r[display] t = %d ." % fid, end="")

            _mat = mat[displayRange[0]: displayRange[1] ,displayRange[0]: displayRange[1]]
            
            if fid == 0:
                cMat, cLabels = getColoredInitMap(_mat)
            else:
                cMat, cLabels = getColoredProceedingMap(_mat, cLabels)

            cMats.append(cMat)
        print()




        imgs = []
        for fid, mat in enumerate(cMats):
            if fid > stopEpoch:
                break
            if fid < startEpoch:
                continue

            _mat = copy.deepcopy(mat)

            _top = L//3 - displayRange[0]
            _bottom = L//3 - displayRange[0] + L//3
            _left = L//3 - displayRange[0]
            _right = L//3 - displayRange[0] + L//3

            if drawGreenRect:
                _mat = cv2.rectangle(copy.deepcopy(_mat), (_left, _top), (_right, _bottom), (0, 191, 0), thickness=1)
            _mat = cv2.putText(_mat, "t=%d" % (fid), (3, 11), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
            
            imgs.append(lib_pil.cv2pil(
                cv2.resize(
                    _mat,
                    (displaySize, displaySize),
                    interpolation=cv2.INTER_NEAREST
                    # interpolation=cv2.INTER_CUBIC
                )
            ))

        # plt.imshow(_mat)

        imgs[0].save(
            result_temp_dp / "colored-result-anime.gif",
            save_all=True,
            append_images=imgs[1:],
            optimize=False,
            duration=1000//fps,
            loop=0
        )
        mlflow.log_artifact(result_temp_dp / "colored-result-anime.gif")

        del _mat


        utility_gif2mp4.trans(result_temp_dp / "colored-result-anime.gif", result_temp_dp / "colored-result-anime.mp4")
        mlflow.log_artifact(result_temp_dp / "colored-result-anime.mp4")



if __name__ == '__main__':
    pass