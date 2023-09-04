import os
import numpy
import matplotlib.pyplot
import copy

def parseLine(line :str):
	x = line[:line.index(",")]
	x = int(x)
	line = line[line.index(",")+1:]
	y = line[:line.index(",")]
	y = int(y)
	line = line[line.index(",")+1:]
	line = line[line.index(",")+1:]
	z = line[:line.index(",")]
	z = int(z)
	line = line[line.index(",")+1:]
	E = int(line)
	return (x, y, z, E)

def getPlot3D(space :numpy.ndarray, name :str):
	'''Shows 3D scatter plot of the input space.'''
	xs, ys, zs = space.nonzero()
	vals = numpy.array([space[xs[i],ys[i],zs[i]] for i in range(len(xs))])
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(projection='3d')
	sctr = ax.scatter(xs, ys, zs, c=vals, cmap="plasma")
	fig.suptitle(name)
	ax.set_xlim(0, 11)
	ax.set_xlabel("$x$")
	ax.set_ylim(0, 13)
	ax.set_ylabel("$y$")
	ax.set_zlim(0, 200)
	ax.set_zlabel("$z$")
	cb = fig.colorbar(sctr, ax=ax)
	cb.set_label("$E$")
	return fig


def getPlotProjections(space :numpy.ndarray, name :str, log :bool = False):
	'''Shows plots of the space projections into the xy, yz and zx planes.'''
	fig, ax = matplotlib.pyplot.subplots(3,1)
	fixed_cmap = copy.copy(matplotlib.cm.get_cmap('gray'))
	fixed_cmap.set_bad((0,0,0))	#fix pixels with zero occurrence - otherwise problem for LogNorm
	if log:
		ax[0].imshow(numpy.sum(space, 0), cmap='gray', norm=matplotlib.colors.LogNorm())
		ax[1].imshow(numpy.sum(space, 1), cmap='gray', norm=matplotlib.colors.LogNorm())
		ax[2].imshow(numpy.sum(space, 2), cmap='gray', norm=matplotlib.colors.LogNorm())
	else:
		ax[0].imshow(numpy.sum(space, 0), cmap='gray')
		ax[1].imshow(numpy.sum(space, 1), cmap='gray')
		ax[2].imshow(numpy.sum(space, 2), cmap='gray')
	ax[0].set_xlabel("$z$")
	ax[0].set_ylabel("$y$")
	ax[1].set_xlabel("$z$")
	ax[1].set_ylabel("$x$")
	ax[2].set_xlabel("$y$")
	ax[2].set_ylabel("$x$")
	fig.suptitle(name)
	return fig


def loadX17Data(track_type :str):
	'''
	Parse X17 data from txt file and yield tuples (event name, 3D event array).

	@track_type: "goodtracks" or "othertracks"
	'''

	PATH = "/data/x17/"
	#for type in ["goodtracks/", "othertracks/"]:
	for f_name in os.listdir(PATH + track_type):
		file = open(PATH + track_type + "/" + f_name, 'r')
		space = numpy.zeros((12,14,208))
		for line in file:
			x, y, z, E = parseLine(line)
			space[x,y,z] = E
		yield (f_name[:-4], space)
