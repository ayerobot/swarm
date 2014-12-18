<<<<<<< HEAD
#ANNULUS METHOD (mostly thought of by Roshan Rao. But Edward Williams helped too.)

#initial world consists of all zeros
#a target point is defined, and annuluses are added (representing points with a known distance from the target but unknown absolute location)
#the number of annuli intersecting that point is the new value at each point
#the maximum value is the most probable location 

from circles_and_such import Point
import numpy as np
import numpy.random as rand

class Annulus:
	def __init__(self, pnt, r1, r2):
		self.point = pnt
		self.r1 = r1 #inner radius
		self.r2 = r2 #outer radius

#defines a world with a data array and a target point
class World:
	def __init__(self, xsize, ysize):
		self.data = np.zeros((xsize, ysize))
		self.target = None 
	def add_annulus(self, annulus1):
		y, x = np.indices(self.data.shape)
		dist = np.sqrt((x - annulus1.point.x)**2 + (y - annulus1.point.y)**2)
		mask = (dist >= annulus1.r1) and (dist <= annulus1.r2)
		self.data[mask] = self.data[mask] + 1
	def probable_locations(self):
		max_val = np.max(self.data)
		return np.where(self.data == max_val)

#generates a world, places random coordinates until the number of probable locations is less than a certain value
def simulate(sizex, sizey,threshold, pos_err_mean, pos_err_stdev):
	wrld = World(sizex, sizey)
	wrld.target = Point(rand.randint(0, wrld.data.shape[0]), rand.randint(0, wrld.data.shape[1]))
	i = 0
	while world.probable_locations().size > threshold:
		#generate random point with distance from target
		other_person = Point(rand.randint(0, wrld.data.shape[0]), rand.randint(0, wrld.data.shape[1]))
		other_person_err = rand.normal(post_err_mean, pos_err_stdev)
		#adding error to distance
		dist_actual = wrld.target.distance_from(other_person)
		dist_err = dist_actual + other_person_err
		new_annulus = Annulus(other_person, dist_actual, dist_err)
		wrld.add_annulus(new_annulus)
		i += 1
		print "People Added: " + i
		print "Probable Locations: " + world.probable_locations().size
=======
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

def iterate(array, xpos, ypos):
	people = generate_people(100, xpos, ypos, array.shape[1], array.shape[0])
	i = 0
	while (i < 30 or check(array) > 200):
		print check(array)
		array = make_annulus(array, people[i][0], people[i][1], people[i][2])
		i += 1
		if (i >= 100):
			break
	print check(array)
	print "num iterations =", i
	return array



>>>>>>> 84a5fb1c31bffab81127561c578350c3cdbf26c5







