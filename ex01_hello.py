

# 5.랜덤 수 생성(Random Number Generating)
# 5.1. 무작위수 1개 얻기(1 Random Number)

rand = tf.random.uniform([1], 0,1)
print(rand)
# > Tensor("random_uniform_2:0", shape=(1,), dtype=float32)

sess = tf.Session()
sess.run(rand)
# > array([0.4512384], dtype=float32)



# 5.2. 무작위수 여러개 얻기(Multi Random Numbers)
# 5.2.1. 균일분포(Uniform Distribution)
rand = tf.random.uniform([4],0,1)
print(rand)
# > Tensor("random_uniform_4:0", shape=(4,), dtype=float32)
sess.run(rand)
# > array([0.14859295, 0.74864423, 0.26292098, 0.14513874], dtype=float32)


# 5.2.2. 정규분포(Normal Distribution)
rand =tf.random.normal([5],0,1)
print(rand)
# > Tensor("random_normal:0", shape=(5,), dtype=float32)
sess.run(rand)
# > array([-2.581728  ,  0.08384032, -0.5727833 ,  0.32536006,  1.6793134 ], dtype=float32)



# 6.1. 확률 변수 X가 정규분포(0,1)을 따를 때 그래프 - 값 변동
# 6.1.1. 라인 그래프(Line Graph)

# tf 1.X
import matplotlib.pyplot as plt

x = range(20)
y = tf.random.normal([20],0,1)
plt.plot(x,y)
plt.show()

# tf 2.X
import matplotlib.pyplot as plt
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

x = range(20)
y = tf.random.normal([20],0,1)
y_sess = sess.run(y)
plt.plot(x,y_sess)
plt.show()
