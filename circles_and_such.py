#source code for a project
#determining if high-accuracy smartphone location measurements can be 
#generated from multiple trilateralization calculations + noisy GPS/loc-services measurements

import math
import numpy as np
import numpy.random as rand

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
	def dist_from(self, other):
		return euclid_distance(self, other)

class Center:
	def __init__(self, pnt, radius):
		self.point = pnt
		self.radius = radius
	def __str__(self):
		return self.point + " radius: " + self.radius

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



