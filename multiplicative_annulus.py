import numpy as np
import numpy.random as rand
from scipy.stats import norm
from random import randint
import matplotlib.pyplot as plt
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
	masks_area = []
	masks.append((dist > radius - sigma_half) & (dist < radius + sigma_half))
	masks_area.append(array[masks[0]].size)
	masks.append((dist > radius - 2*sigma_half) & (dist < radius + 2*sigma_half) & ~masks[0])
	masks_area.append(array[masks[1]].size)
	masks.append((dist > radius - 3*sigma_half) & (dist < radius + 3*sigma_half) & ~(masks[0] | masks[1]))
	masks_area.append(array[masks[2]].size)
	masks.append((dist > radius - 4*sigma_half) & (dist < radius + 4*sigma_half) & ~(masks[0] | masks[1] | masks[2]))
	masks_area.append(array[masks[3]].size)
	masks.append(~(masks[0] | masks[1] | masks[2] | masks[3]))
	masks_area.append(array[masks[4]].size)
	for i, mask in enumerate(masks):
		array[mask] = array[mask] * probs[i] / masks_area[i]
	return array

#returns n-dimensional euclidian distance between two array-like points
#input arrays have same shape
def euclid(arr1, arr2):
	if arr1.shape == arr2.shape:
		total = 0
		for i, val in enumerate(arr1):
			total += (val - arr2[i])**2
		return np.sqrt(total)
	else:
		raise TypeError("shapes do not match")


def generate_probability_annulus(shape, x, y, measured_dist, sigma):
	yind, xind = np.indices(shape)
	dist = np.sqrt((xind - x)**2 + (yind - y)**2)
	prob_array = norm.pdf(dist, measured_dist, sigma)
	return prob_array/prob_array.sum()

#generates probability annuli until the limit has been reached
#returns an averaged probability distribution 
def iterate_until(shape, xtarg, ytarg, low_err, high_err, limit):
	i = 0
	total = np.zeros(shape)
	while(i < limit):
		#generates random location + distance
		x = rand.randint(0, shape[0])
		y = rand.randint(0, shape[1])
		dist = euclid(np.array([x, y]), np.array([xtarg, ytarg])) # calculates relative distance
		sigma = rand.randint(low_err, high_err) # creates random noise
		dist += np.random.normal(0, sigma)
		new_annulus  = generate_probability_annulus(shape, x, y, dist, sigma)
		i += 1
		total += new_annulus
	return total/limit

def generate_people(num, xpos, ypos, xmax, ymax):
	people = np.zeros((num, 3))
	people[:,0] = np.random.randint(0, xmax, (1, num))
	people[:,1] = np.random.randint(0, ymax, (1, num))
	people[:,2] = np.array([dist(people[i,0], people[i,1], xpos, ypos) + np.random.normal(0, sigma) for i in range(num)])
	return people

def iterate(array, xpos, ypos, num):
	people = generate_people(num, xpos, ypos, array.shape[1], array.shape[0])
	i = 0
	while (i < num or check(array) > 200):
		print check(array)
		print "Max probability: " +  str(array.max())
		print "Sum: " + str(array.sum())
		array = make_annulus(array, people[i][0], people[i][1], people[i][2])
		i += 1
		#if (i >= num):
		#	break
	print check(array)
	print "num iterations =", i
	return array

#test function, needs to be expanded
def test2(num):
	shape = (2000, 2000)
	targx = 1000
	targy = 1000
	result = iterate_until(shape, targx, targy, 10, 30, num)
	random_probability = 1.0/(shape[0]*shape[1])
	max_prob = result.max()
	print "probability of the target at a random spot with no reference: " + str(random_probability)
	print "max probability with " + str(num) + " reference points: " +  str(max_prob)
	print "probability increase: " + str(max_prob/random_probability)
	#get number of elements that are 90 percent or higher of the maximum probability:
	max_prob_err = max_prob*0.90
	close_points = np.where(result > max_prob_err)[0]
	print "total number of points: " + str(shape[0]*shape[1])
	print "number of points with 90 percent of max probability or higher: " + str(close_points.size)
	return result

def test():
	ones = np.ones((2000, 2000))
	iterate(ones/ones.size, 1000, 1000, 30)

#testing the normal distribution methods
def fun_with_normal(xsize, ysize):
	ones = np.ones((xsize, ysize))
	arr = ones/ones.size
	mean_dist = 30 #the measured distance from the reference coordinate
	sigma = 10
	ref_coord = (50, 50)
	x, y = ref_coord #clean this part up
	yind, xind = np.indices(arr.shape)
	dist_arr = np.sqrt((xind - x)**2 + (yind - y)**2)
	prob_array = norm.pdf(dist_arr, mean_dist, sigma)
	#correct the probability array by dividing by its SUM, not its size! 
	#pretty sure that's what roshan meant but I misinterpreted it
	prob_array_corrected = prob_array/prob_array.sum()
	return prob_array_corrected

def graph_results(ref_num, arr):
	plt.pcolormesh(arr)
	plt.xlabel("x coordinate")
	plt.ylabel("y coordinate")
	plt.title('location probability with ' + str(ref_num) + ' reference points')
	plt.colorbar()
	plt.show()

if __name__ == "__main__":
	num = 4
	arr = test2(num)
	graph_results(num, arr)








