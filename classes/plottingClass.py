import matplotlib
import numpy
import copy
from classes import modelWrapperClass

class Plotting:
	'''
	Class wrapping methods for plotting.
	'''

	@staticmethod
	def plotEvent(noisy, reconstruction, classificated = None, are_data_experimental = None, model_name = '', axes=[0,1,2], use_log :bool = False, event_name :str = None):
		'''
		Create plot of track reconstruction.
		@noisy ... Noisy event tensor.
		@reconstruction ... Tensor of event reconstructed by model.
		@classificated ... Event after threshold classification. Default is None, which skips this plot.
		@are_data_experimental ... False iff the event is from the generated dataset.
		@model_name ... Name of the model, which will be displayed in the plot.
		@axes ... List of plotted projection axes.
		@use_log ... If True, use log scale.	
		'''
		
		if classificated is None:
			fig, ax = matplotlib.pyplot.subplots(len(axes), 2)
		else:
			fig, ax = matplotlib.pyplot.subplots(len(axes), 3)

		x_labels = ['z', 'z', 'y']
		y_labels = ['y', 'x', 'x']

		fixed_cmap = copy.copy(matplotlib.cm.get_cmap('gray'))
		fixed_cmap.set_bad((0,0,0))	#fix pixels with zero occurrence - otherwise problem for LogNorm
		norm = matplotlib.colors.LogNorm(vmin=0, vmax=1) if use_log else matplotlib.colors.PowerNorm(1, vmin=0, vmax=1)

		for i in range(len(axes)):
			axis = axes[i]
			ax[i][0].set_title("Noisy")
			ax[i][0].imshow(numpy.sum(noisy, axis=axis), cmap=fixed_cmap, norm=norm )
			ax[i][0].set_xlabel(x_labels[axis])
			ax[i][0].set_ylabel(y_labels[axis])
			ax[i][1].set_title("Raw Reconstruction")
			ax[i][1].imshow(numpy.sum(reconstruction, axis=axis), cmap=fixed_cmap, norm=norm )
			ax[i][1].set_xlabel(x_labels[axis])
			ax[i][1].set_ylabel(y_labels[axis])

		if classificated is not None:
			for i in range(len(axes)):
				axis = axes[i]
				ax[i][2].set_title("After Threshold")
				ax[i][2].imshow(numpy.sum(classificated, axis=axis), cmap=fixed_cmap, norm=norm )
				ax[i][2].set_xlabel(x_labels[axis])
				ax[i][2].set_ylabel(y_labels[axis])
			
		title = "Reconstruction of "
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		title += "Data "
		if event_name is not None:	title += "(" + event_name + ") "
		title += "by Model " + model_name
		fig.suptitle(title)
	
	@staticmethod
	def getPlotEventOneAxis(noise_event :numpy.ndarray, nonNN_event :numpy.ndarray, NN_event :numpy.ndarray, axis :int, event_name :str = None, cmap :str="Greys"):
		'''
		Plot projection of @noise_data, its NN and non-NN reconstruction (@NN_event and @nonNN_event, respectively) in specified @axis. 
		'''

		x_labels = ['z', 'z', 'y']
		y_labels = ['y', 'x', 'x']

		fig, ax = matplotlib.pyplot.subplots(3)
		
		cmap = matplotlib.pyplot.get_cmap(cmap)
		cmap.set_under('cyan')
		eps = 1e-8

		ax[0].set_title("Noisy " + event_name)
		ax[0].imshow(numpy.sum(noise_event, axis=axis), cmap=cmap, vmin=eps)
		ax[1].set_title("non-NN Reconstruction")
		ax[1].imshow(numpy.sum(nonNN_event, axis=axis), cmap=cmap, vmin=numpy.min([eps, numpy.max(nonNN_event)]))
		ax[2].set_title("NN Reconstruction")
		ax[2].imshow(numpy.sum(NN_event, axis=axis), cmap=cmap, vmin=numpy.min([eps, numpy.max(NN_event)]))
		for i in range(3):
			ax[i].set_xlabel(x_labels[axis])
			ax[i].set_ylabel(y_labels[axis])

	@staticmethod
	def plotRandomData(modelAPI :modelWrapperClass.ModelWrapper, noise_data :numpy.ndarray, are_data_experimental :bool = None, axes :list = [0,1,2], use_log :bool = False):
		'''
		Plot @model's reconstruction of random events from @noise_data. If @threshold is specified, plot also the final classification after applying @threshold to reconstruciton.
		'''

		while True:
			index = numpy.random.randint(0, len(noise_data))
			noisy = noise_data[index]
			reconstr = modelAPI.evaluateSingleEvent(noisy)

			if modelAPI.hasThreshold():
				classif = modelAPI.classify(reconstr)
				Plotting.plotEvent(noisy, reconstr, classif, are_data_experimental, modelAPI.name, axes = axes, use_log = use_log)
			else:
				Plotting.plotEvent(noisy, reconstr, None, are_data_experimental, modelAPI.name, axes = axes, use_log = use_log)
			matplotlib.pyplot.show()
			if input("Enter 'q' to stop plotting (or anything else for another plot):") == "q":	break
	
	@staticmethod
	def getPlot3D(modelAPI :modelWrapperClass.ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None, event_name :str = None):
		'''
		Return 3D plot of @noise_event and its reconstruction by @model.
		@model ... Keras model reconstructing track in this plot.
		@noise_event ... One noisy event tensor.
		@are_data_experimental ... False iff the event is from generated dataset.
		@model_name ... Name of the model which will be displayed in the plot.
		@threshold ... Classification threshold for the model.
		@rotation ... Float triple specifying the plot should be rotated.
		'''

		fig = matplotlib.pyplot.figure(figsize=matplotlib.pyplot.figaspect(0.5))
		ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		title = ""
		if are_data_experimental:	title += "Experimental "
		elif are_data_experimental is False:	title += "Generated "
		if event_name is not None:	title += "(" + event_name + ") "
		title += "Data"
		M = numpy.max(noise_event)
		if M == 0:	M = 1
		scaleSize = (lambda x: 100*x/M + 30) if are_data_experimental else (lambda x: 150*x + 50)
		sctr = Plotting.plot3DToAxis(noise_event, ax1, title, scaleSize)

		reconstr_event = modelAPI.evaluateSingleEvent( noise_event / (numpy.max(noise_event) if numpy.max(noise_event) != 0 else 1) )
		classificated_event = modelAPI.classify(reconstr_event)
		ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		title = "Reconstruction and Threshold Classification\n"
		title += "by Model " + modelAPI.name
		Plotting.plot3DToAxis(classificated_event, ax2, title)

		cb = fig.colorbar(sctr, ax=[ax1, ax2], orientation="horizontal")
		cb.set_label("$E$")

		return fig, ax1, ax2

	def plot3DToAxis(event :numpy.ndarray, ax, title :str = "", z_cut = (0,200), cmap="copper_r"):
		'''
		Create 3D plot of @event on specified matplotlib axis @ax.
		@scaleSize ... Function that scales scatter point size based on the corresponding value.
		@z_cut ... (z_low, z_max) limits of z axis.
		'''

		def getOffset(val, max_E):
			rel = val/max_E
			return (5 - int(numpy.sqrt(30*rel)) )

		max_E = numpy.max(event)
		#scaleSize = lambda x: 0.3*(20*(x/max_E) + 20)
		scaleSize = lambda x: 6

		xs, ys, zs = event.nonzero()
		vals = [event[xs[i],ys[i],zs[i]] for i in range(len(xs))]

		xs_plot, ys_plot, zs_plot, vals_plot = [], [], [], []
		for n in range(len(xs)):
			val = vals[n]
			offset = getOffset(val, max_E)
			for i in range(offset,12-offset):
				for j in range(offset,14-offset):
					xs_plot.append(10*xs[n] + i)
					ys_plot.append(10*ys[n] + j)
					zs_plot.append(zs[n])
					vals_plot.append(val)
		#xs_plot = 10*xs + 5
		#ys_plot = 10*ys + 5
		#zs_plot = zs
		

		sctr = ax.scatter(xs_plot, ys_plot, zs_plot, c=vals_plot, cmap = matplotlib.pyplot.get_cmap(cmap), marker="s", edgecolors="black", linewidths=0.1, s=scaleSize(vals_plot), vmin=0)
		ax.set_xlim(0, 110)
		ax.set_xlabel("$x$")
		ax.set_ylim(0, 130)
		ax.set_ylabel("$y$")
		ax.set_zlim(*z_cut)
		ax.set_zlabel("$z$")
		ax.set_title(title)
		ax.set_box_aspect((14, 14, 20))
		return sctr

	def animation3D(path :str, modelAPI :modelWrapperClass.ModelWrapper, noise_event :numpy.ndarray, are_data_experimental :bool = None):
		'''
		Create and save gif of rotating 3D plot.
		'''

		fig, ax1, ax2 = Plotting.getPlot3D(modelAPI, noise_event, are_data_experimental)

		def run(i):	
			ax1.view_init(0,i,0)
			ax2.view_init(0,i,0)

		anim = matplotlib.animation.FuncAnimation(fig, func=run, frames=360, interval=20, blit=False)
		anim.save(path, fps=30, dpi=200, writer="pillow")


	def plotTileDistribution(data :numpy.ndarray, modelAPI :modelWrapperClass.ModelWrapper):
		'''
		Create histogram of tile z coordinates distribution 
		'''

		classified = modelAPI.classify( modelAPI.evaluateBatch(data) )
		fig, ax = matplotlib.pyplot.subplots(1)
		counts_raw = numpy.sum(numpy.where(data>0.000001,1,0), axis=(0,1,2))
		counts_rec = numpy.sum(classified, axis=(0,1,2))
		ax.hist(x=[i for i in range(208)], bins=69, weights=counts_raw, label="Noisy", histtype="step")
		ax.hist(x=[i for i in range(208)], bins=69, weights=counts_rec, label="Reconstructed")
		ax.set_title("Distribution of X17 tile z coordinates after reconstruction by model " + modelAPI.name)
		ax.set_xlabel("z")
		ax.set_ylabel("#")
		ax.legend()


	def plot3D(event :numpy.ndarray, title :str = "", eps :float = 1e-6, azimuth :float = None, elev :float = None, cmap="copper_r", z_cut=(0,200)):
		
		def upsample(matrix):
			return numpy.repeat(numpy.repeat( numpy.copy(matrix) , 10, axis=0), 11, axis=1)[:100,:120,:200]

		event = numpy.copy(event)
		event = numpy.where(event > eps, event, 0)
		event_upsample = numpy.copy(event)
		event_upsample = upsample(event_upsample)
		
		fig, ax = matplotlib.pyplot.subplots(1)
		ax.remove()
		ax = fig.add_subplot(1,1,1,projection='3d')

		Plotting.plot3DToAxis(event, ax, title, z_cut=z_cut, cmap=cmap)
		if azimuth is not None and elev is None:
			ax.view_init(azim=azimuth)
		if elev is not None and azimuth is None:
			ax.view_init(elev=elev)
		if azimuth is not None and elev is not None:
			ax.view_init(azim=azimuth, elev=elev)
		
		ax.tick_params(axis="x", pad=-2)
		ax.set_xlabel("x (mm)", labelpad=-6)
		ax.tick_params(axis="y", pad=-2)
		ax.set_ylabel("y (mm)", labelpad=-4)
		ax.set_zlabel("z (mm)")
		#fig.tight_layout()
		return fig, ax

	def plot2DAnd3D(event :numpy.ndarray, title :str = "", eps :float = 1e-6, azimuth :float = None, elev :float = None):
		cmap = matplotlib.pyplot.get_cmap("Greys")
		cmap.set_under('cyan')

		def upsample(matrix):
			return numpy.repeat(numpy.repeat( numpy.copy(matrix) , 10, axis=0), 11, axis=1)[:100,:120,:200]

		fig, ax = matplotlib.pyplot.subplots(2,2, gridspec_kw={"height_ratios": [20,10], "width_ratios": [10,20]}, layout="constrained")
		fig.suptitle(title)
		#for i in [0,1]:
		#	for j in [0,1]:
		#		ax[i,j].set_aspect("equal")

		event = numpy.copy(event)
		event = numpy.where(event > eps, event, 0)
		event_upsample = numpy.copy(event)
		event_upsample = upsample(event_upsample)
		
		zx = numpy.sum(event_upsample, 1)
		xz = numpy.transpose(zx)
		ax[0,0].imshow( xz, origin="lower", cmap=cmap, vmin=1e-4 )
		#ax[0,0].set_xlabel("x")
		ax[0,0].set_ylabel("z")
		ax[0,0].set_title("xz projection")

		yx = numpy.sum(event_upsample, 2)
		xy = numpy.transpose(yx)
		ax[1,0].imshow( xy, origin="lower", cmap=cmap, vmin=1e-4 )
		ax[1,0].set_xlabel("x")
		ax[1,0].set_ylabel("y")
		ax[1,0].set_title("xy projection")
		ax[1,0].set_aspect(12/14)

		zy = numpy.sum(event_upsample, 0)
		ax[1,1].imshow( zy, origin="lower", cmap=cmap, vmin=1e-4 )
		ax[1,1].set_xlabel("z")
		#ax[1,1].set_ylabel("y")
		ax[1,1].set_title("zy projection")

		ax[0,1].remove()
		ax[0,1]=fig.add_subplot(2,2,2,projection='3d')

		Plotting.plot3DToAxis(event, ax[0,1], "3D")
		if azimuth is not None and elev is None:
			ax[0,1].view_init(azim=azimuth)
		if elev is not None and azimuth is None:
			ax[0,1].view_init(elev=elev)
		if azimuth is not None and elev is not None:
			ax[0,1].view_init(azim=azimuth, elev=elev)
		

		ax[0,1].tick_params(axis="x", pad=-2)
		ax[0,1].set_xlabel("x", labelpad=-6)
		ax[0,1].tick_params(axis="y", pad=-2)
		ax[0,1].set_ylabel("y", labelpad=-4)
		
		#fig.tight_layout()
		return fig, ax