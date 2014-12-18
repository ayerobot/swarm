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







