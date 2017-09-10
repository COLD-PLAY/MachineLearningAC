import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# plt.plot([2, 3, 4, 5], [4, 5, 6, 7], 'r-')
# # axis([xmin,xmax,ymin,ymax]) 限制坐标轴
# plt.axis([0, 6, 0, 20])

# plt.ylabel('some numbers')
# plt.show()

# t = np.arange(0., 5., 0.2)
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^', linewidth=2.0)
# plt.show()

# def f(t):
# 	return np.exp(-t) * np.cos(2*np.pi*t)

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)

# plt.figure(1)
# plt.subplot(211)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# plt.figure(2)
# plt.plot([1, 2, 3, 4])

# plt.figure(1)
# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

# plt.title('easy as 1, 2, 3')
# plt.show()

# x = np.linspace(0.0, 5.0, 50)
# y1 = np.sin(x)
# y2 = np.exp(x) * np.cos(x)

# figures = [plt.figure(1), plt.figure(2)]

# plt.figure(1)
# axeses = [plt.subplot(221), plt.subplot(224)]

# plt.subplot(221)
# lines = plt.plot(x, y1, x, y2)
# plt.subplot(224)
# plt.plot(x)

# plt.setp(figures, facecolor='m')
# plt.setp(axeses, facecolor='w')
# plt.setp(lines, linestyle='dashdot')

# plt.figure(2)
# plt.plot(y1*y2)

# plt.show()

# Fixing random state for reproducibility
# np.random.seed(19680801)

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)

# # the histogram of the data
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)

# plt.show()

# ax = plt.subplot(111)

# t = np.arange(0.0, 5.0, 0.01)
# s = np.cos(2 * np.pi * t)

# line, = plt.plot(t, s, lw=2)

# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
# 	arrowprops=dict(facecolor='black', shrink=0.05),
# 	)

# plt.ylim(-2, 2)
# # plt.axis([0, 5, -2, 2])

# plt.show()

from scipy import misc
def rgb2gray(rgb):
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

seven = mpimg.imread('seven.jpg')
# plt.axis('off')

# plt.imshow(seven)
# plt.show()

# seven_gray = rgb2gray(seven)
# seven_32 = misc.imresize(seven, 0.025)

# plt.imshow(seven_32)
# plt.show()
# print(seven_32.shape)

# figure = plt.figure()
# ax = figure.add_subplot(111)
# ax.axis([0.0, 32.0, 0.0, 32.0])


seven_32 = mpimg.imread('seven_32.jpg')
# plt.imshow(seven_32)
# plt.show()

figure = plt.figure()
ax = figure.add_subplot(111)
ax.axis([0, 32, 0, 32])

c = 0

for i in range(32):
	for j in range(32):
		if sum(seven_32[i, j, :]) <= 750:
			c += 1
			# print(sum(seven_32[i, j, :]))
			# print(seven_32[i, j, :])
			print(i, j)
			ax.plot(j, 32 - i, 'ro')
		else:
			ax.plot(j, 32 - i, 'ko')

print('totla %d' % c)
plt.show()