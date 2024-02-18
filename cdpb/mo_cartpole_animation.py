import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mo_cdpb import CartPole
import neat
import numpy as np

# ニューラルネットワークの設定
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'cdpb_config-feedforward.py')  # あなたのconfigファイルへのパスをここに入力してください

# ニューラルネットワークの作成
genome = neat.DefaultGenome(0)  # キーを適切なものに設定する
genome.configure_new(config.genome_config)

net = neat.nn.FeedForwardNetwork.create(genome, config)

# CartPoleのインスタンスの作成
cart_pole = CartPole()

# 各タイムステップでのカートとポールの位置を計算
data = []
for i in range(100):
    x = [0.0, 0.0, np.pi*1/180, 0.0]
    output = net.activate(x)
    data.append(cart_pole.generate_data(output))

# プロットの初期設定
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

# アニメーションの初期化
def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 2)
    line.set_data([], [])
    return line,

# アニメーションフレームの更新
def update(frame):
    x, y = frame[0], np.sin(frame[2])
    line.set_data(x, y)
    return line,

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=data, init_func=init, blit=True)

# アニメーションの表示
plt.show()
