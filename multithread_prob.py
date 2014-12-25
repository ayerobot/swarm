from multiplicative_annulus import euclid, generate_probability_annulus, add_error
import numpy as np
import numpy.random as rand
import threading

dist_func = lambda x1, y1, x2, y2: np.sqrt((x1 - x2)**2 + (y1 - y2)**2) # returns distance between two points

class IterThread(threading.Thread):
	def __init__(self, threadID, array, arrayLock, x, y, dist, sigma):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.array = array
		self.arrayLock = arrayLock
		self.x = x
		self.y = y
		self.dist = dist
		self.sigma = sigma
	def run(self):
		new_annulus = generate_probability_annulus(self.array.shape, 
			self.x, self.y, self.dist, self.sigma)
		self.arrayLock.acquire()
		self.array += new_annulus
		self.arrayLock.release()

class TestThread(threading.Thread):
	def __init__(self, threadID, array, arrayLock, num):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.array = array
		self.arrayLock = arrayLock
		self.num = num
	def run(self):
		dist = test(self.num)
		self.arrayLock.acquire()
		self.array.append(dist)
		self.arrayLock.release()


def iterate_until(shape, xtarg, ytarg, low_err, high_err, limit):
	total = np.zeros(shape)
	lock = threading.Lock()
	threads = []
	for i in range(limit):
		x = rand.randint(0, shape[0])
		y = rand.randint(0, shape[1])
		dist = euclid(np.array([x, y]), np.array([xtarg, ytarg]))
		sigma = rand.randint(low_err, high_err)
		x, y = add_error(x, y, sigma)
		thread = IterThread(i, total, lock, x, y, dist, sigma)
		thread.start()
		threads.append(thread)

	for t in threads:
		t.join()

	return total / limit

def test(num):
	targx, targy = (1000, 1000)
	#sigma = rand.randint(10, 30)
	# for test purposes make sigma constant at max_error
	sigma = 30
	measuredX, measuredY = add_error(targx, targy, sigma)
	original = generate_probability_annulus((2000, 2000), measuredX, measuredY, 0, sigma)
	result = iterate_until((2000, 2000), targx, targy, 10, 30, num)

	original_center = np.where(original == original.max())
	new_center = np.where(result == result.max())
	#print "Original center:", original_center
	#print "New center:", new_center
	if (len(original_center[0]) == 1 and len(new_center[0]) == 1):
		origy, origx = original_center[0][0], original_center[1][0]
		newy, newx = new_center[0][0], new_center[1][0]
		orig_dist = dist_func(origx, origy, targx, targy)
		new_dist = dist_func(newx, newy, targx, targy)
		#print "Original distance:", orig_dist
		#print "New distance:", new_dist
		#print "\n"
		return new_dist
	else:
		#print "\n"
		return None

def long_test(num_trials, num_annuli):
	array = []
	arrayLock = threading.Lock()
	threads = []
	for i in range(num_trials):
		print "Starting thread", i
		thread = TestThread(i, array, arrayLock, num_annuli)
		thread.start()
		threads.append(thread)

	for t in threads:
		t.join()
		print "thread", t.threadID, "finished"

	return array

