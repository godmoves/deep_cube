import random

import face
import cubie
import time


def reverse_move(move):
    return 3 * (move // 3) + 2 - (move % 3)


def generate_training_data(file_path, games=1, moves=12, verbose=False):
    cc = cubie.CubieCube()
    print('Generating {} ...'.format(file_path))
    with open(file_path, 'w+') as f:
        for g in range(games):
            move_list = []
            for m in range(moves):
                move = random.randint(0, 17)
                move_list.append(move)
                cc.multiply(cubie.moveCube[move])
                f.write(cc.to_facelet_cube().to_string())
                f.write('\n{}\n{}\n'.format(reverse_move(move), m + 2))
            if verbose:
                print(move_list)


games_per_file = 1000
total_file_num = 1

begin_time = time.time()
for i in range(total_file_num):
    generate_training_data('./train/gen2_{}'.format(i), games=games_per_file)
end_time = time.time()
print('Generate {} files in {:.4f}s'.format(total_file_num, end_time - begin_time))
