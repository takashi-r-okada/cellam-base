# coding: utf-8

import moviepy.editor as mp
import sys
from pathlib import Path


def trans(srcFP, tgtFP):

    #gif動画ファイルの読み込み
    movie_file=mp.VideoFileClip(str(srcFP))
    
    #mp4動画ファイルの保存
    movie_file.write_videofile(str(tgtFP))
    movie_file.close()

if __name__ == "__main__":
    srcFP = Path(sys.argv[1])
    tgtFP = srcFP.parent / (srcFP.stem + ".mp4")
    trans(srcFP, tgtFP)
