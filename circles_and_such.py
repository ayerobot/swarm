#source code for a project
#determining if high-accuracy smartphone location measurements can be 
#generated from multiple trilateralization calculations + noisy GPS/loc-services measurements

import math
import numpy as np
import np.random as rand

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __add__(self, other):
		return point(self.x + other.x, self.y + other.y)
	def __sub__(self, other):
		return point(self.x - other.x, self.y - other.y)
	def __str__(self):
		return "(" + self.x + "," + self.y + ")"

class Center:
	def __init__(self, pnt, radius):
		self.point = pnt
		self.radius = radius
	def __str__(self):
		return self.point + " radius: " + self.radius

class Annulus:
	def __init__(self, pnt, r1, r2):
		self.point = pnt
		self.r1 = r1 #inner radius
		self.r2 = r2 #outer radius

def euclid_distance(point1, point2):
	return sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


#credit for algorithm goes to Paul Bourke
#http://paulbourke.net/geometry/circlesphere/
def two_circles(cent1, cent2):
	#checking if intersection is possible
	d = euclid_distance(cent1.point, cent2.point)
	if d > (cent1.radius + cent2.radius):
		return -1
	elif d < abs(cent1.radius - cent2.radius):
		return -1
	elif (d == 0) and (cent1.radius == cent2.radius):
		#infinite solutions
		return -1

	a = (cent1.radius**2 - cent2.radius**2 + d**2)/(2*d)
	#I'll finish this later, might not need it?
	p2 = 0

#ANNULUS METHOD (mostly thought of by Roshan Rao. But Edward Williams helped too.)

#initial world consists of all zeros
#a target point is defined, and annuluses are added (representing points with a known distance from the target but unknown absolute location)
#the number of annuli intersecting that point is the new value at each point
#the maximum value is the most probable location 

def add_annulus(old_world, annulus1):
	y, x = np.indices(old_world.shape)
	dist = np.sqrt((x - annulus1.point.x)**2 + (y - annulus1.point.y)**2)
	mask = (dist >= annulus1.r1) and (dist <= annulus1.r2)
	old_world[mask] = old_world[mask] + 1
	return old_world

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
def simulate(sizex, sizey,threshold, pos_err):
	wrld = World(sizex, sizey)
	random_pos = Point(rand.randint(0, wrld.shape[0]), rand.randint(0, wrld.shape[1]))
	wrld.target = random_pos
	while world.probable_locations().size > threshold:
		#generate random point with distance from target





