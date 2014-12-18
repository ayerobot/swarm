import numpy as np
from random import randint
"""
Assume noise is normally distributed. Noise value is margin of error for a
given position with 95 percent confidence. Therefore noise should be +/- 2sigma. 
Probs contains the probability that the actual location will be contained in a 
given area, where area is broken into segments of size sigma/2 (with the exception of the
final segment, which is on (2sigma, infinity)).
"""
noise = 100
sigma = noise / 2.
sigma_half = noise / 4.
probs = [0.383, 0.2996, 0.1838, 0.088, 0.0456]
#np.random.normal(mu, sigma, shape)

check = lambda array: len(np.where(array == array.max())[0]) # returns number of locations where array == array.max()
dist = lambda x1, y1, x2, y2: np.sqrt((x1 - x2)**2 + (y1 - y2)**2) # returns distance between two points

def make_annulus(array, x, y, radius):
	yind, xind = np.indices(array.shape)
	dist = np.sqrt((xind - x)**2 + (yind - y)**2)
	masks = []
	masks.append((dist > radius - sigma_half) & (dist < radius + sigma_half))
	masks.append((dist > radius - 2*sigma_half) & (dist < radius + 2*sigma_half) & ~masks[0])
	masks.append((dist > radius - 3*sigma_half) & (dist < radius + 3*sigma_half) & ~(masks[0] | masks[1]))
	masks.append((dist > radius - 4*sigma_half) & (dist < radius + 4*sigma_half) & ~(masks[0] | masks[1] | masks[2]))
	masks.append(~(masks[0] | masks[1] | masks[2] | masks[3]))
	for i, mask in enumerate(masks):
		array[mask] = array[mask] * probs[i]
	return array

def generate_people(num, xpos, ypos, xmax, ymax):
	people = np.zeros((num, 3))
	people[:,0] = np.random.randint(0, xmax, (1, num))
	people[:,1] = np.random.randint(0, ymax, (1, num))
	people[:,2] = np.array([dist(people[i,0], people[i,1], xpos, ypos) + np.random.normal(0, sigma) for i in range(num)])
	return people

def iterate(array, xpos, ypos, num):
	people = generate_people(num, xpos, ypos, array.shape[1], array.shape[0])
	i = 0
	while (i < 30 or check(array) > 200):
		print check(array)
		array = make_annulus(array, people[i][0], people[i][1], people[i][2])
		i += 1
		if (i >= num):
			break
	print check(array)
	print "num iterations =", i
	return array










