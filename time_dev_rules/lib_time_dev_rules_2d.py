# coding: utf-8

import numpy as np

# --------------------------------------------------
# 時間発展規則の定義
# --------------------------------------------------

class Rules():

    def life_game(state: np.ndarray):
        '''
        ライフゲームの規則。

        [引数]
        - state: [3, 3] array
        '''

        assert state.shape[0] == 3
        assert state.shape[1] == 3
        assert np.max(state) <= 1
        assert np.min(state) >= 0


        if state[1, 1] == 0:
            if np.sum(state) == 3:
                return 1
            else:
                return 0
        else:
            if np.sum(state) in [2+1, 3+1]:
                return 1
            else:
                return 0
            


    def life_game_with_chain(state: np.ndarray):
        '''
        ライフゲームの規則。但し、オセロのように両端を挟まれた場合には中心セルの生存を存続させる。

        [引数]
        - state: [3, 3] array
        '''

        assert state.shape[0] == 3
        assert state.shape[1] == 3
        assert np.max(state) <= 1
        assert np.min(state) >= 0


        if state[1, 1] == 0:
            if np.sum(state) == 3:
                return 1
            elif np.sum(state) == 2:
                if state[0,0] + state[2,2] == 2:
                    return 1
                elif state[0,1] + state[2,1] == 2:
                    return 1
                elif state[0,2] + state[2,0] == 2:
                    return 1
                elif state[1,0] + state[1,2] == 2:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            if np.sum(state) in [2+1, 3+1]:
                return 1
            else:
                return 0
        

# --------------------------------------------------
# 時間発展規則を得るための名称の定義
# --------------------------------------------------

TIME_DEV_RULES_2D = {
    "life-game": Rules.life_game,
    "life-game-with-chain": Rules.life_game_with_chain,
}