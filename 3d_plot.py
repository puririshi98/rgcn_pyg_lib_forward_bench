import matplotlib.pyplot as plt
pyg_lib_times = {(2, 4): 5.8836936950683594e-05, (2, 8): 6.75821304321289e-05, (2, 16): 8.696556091308594e-05, (2, 32): 0.00011962413787841797, (2, 64): 0.00019196987152099608, (2, 128): 0.0003341817855834961, (2, 256): 0.0006355857849121094, (4, 8): 7.464885711669922e-05, (4, 16): 9.210586547851562e-05, (4, 32): 0.00012973785400390624, (4, 64): 0.0002005290985107422, (4, 128): 0.00035129547119140626, (4, 256): 0.0006489849090576172, (8, 16): 0.00010851383209228515, (8, 32): 0.00014110088348388673, (8, 64): 0.00021492958068847657, (8, 128): 0.0003680753707885742, (8, 256): 0.0006784868240356446, (16, 32): 0.00016693115234375, (16, 64): 0.0002422046661376953, (16, 128): 0.00039249420166015625, (16, 256): 0.0006949853897094727, (32, 64): 0.00029033660888671875, (32, 128): 0.00044193267822265623, (32, 256): 0.0007503414154052734, (64, 128): 0.0005375242233276367, (64, 256): 0.0008515548706054687, (128, 256): 0.0010420703887939454}
vanilla_times = {(2, 4): 4.883289337158203e-05, (2, 8): 8.893966674804688e-05, (2, 16): 0.00017212390899658204, (2, 32): 0.0003407812118530273, (2, 64): 0.000681924819946289, (2, 128): 0.0013367414474487304, (2, 256): 0.002787208557128906, (4, 8): 0.00010905265808105468, (4, 16): 0.00018270015716552733, (4, 32): 0.00035500049591064454, (4, 64): 0.0006937789916992188, (4, 128): 0.0013601446151733398, (4, 256): 0.0027700042724609374, (8, 16): 0.00019781589508056642, (8, 32): 0.0003643512725830078, (8, 64): 0.0007050371170043945, (8, 128): 0.0014141845703125, (8, 256): 0.0028091716766357423, (16, 32): 0.000390019416809082, (16, 64): 0.00074371337890625, (16, 128): 0.0014589643478393555, (16, 256): 0.002828555107116699, (32, 64): 0.0007866382598876953, (32, 128): 0.0014922237396240235, (32, 256): 0.002868914604187012, (64, 128): 0.0015906000137329102, (64, 256): 0.003030433654785156, (128, 256): 0.0031949520111083985}
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
pyg_x, pyg_y, pyg_z = [], [], []
vanilla_x, vanilla_y, vanilla_z = [], [], []
for key in pyg_lib_times.keys():
	pyg_x.append(key[0])
	vanilla_x.append(key[0])
	pyg_y.append(key[1])
	vanilla_y.append(key[1])
	pyg_z.append(pyg_lib_times[key])
	vanilla_z.append(vanilla_times[key])
ax.scatter(pyg_x, pyg_y, pyg_z, label='pyg_lib fwd pass time')
ax.scatter(vanilla_x, vanilla_y, vanilla_z, label='vanilla fwd pass time')
ax.set_xlabel('# Node Types')
ax.set_ylabel('# Edge Types')
ax.set_zlabel('Forward Pass Time (s)')
ax.legend()
plt.show()

#animate:
# slow start
for i, angle in enumerate(list(range(1,20,1))):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(pyg_x, pyg_y, pyg_z, label='pyg_lib fwd pass time')
	ax.scatter(vanilla_x, vanilla_y, vanilla_z, label='vanilla fwd pass time')
	ax.set_xlabel('# Node Types')
	ax.set_ylabel('# Edge Types')
	ax.set_zlabel('Forward Pass Time (s)')
	ax.legend()
	ax.view_init(30,270.0+angle/4.0)
	filename='3dplot/bench_step'+'x'*(i)+'.png'
	plt.savefig(filename)
count = i
# fast through midrange
for i, angle in enumerate(range(275,355,2)):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(pyg_x, pyg_y, pyg_z, label='pyg_lib fwd pass time')
	ax.scatter(vanilla_x, vanilla_y, vanilla_z, label='vanilla fwd pass time')
	ax.set_xlabel('# Node Types')
	ax.set_ylabel('# Edge Types')
	ax.set_zlabel('Forward Pass Time (s)')
	ax.legend()
	last_angle = 30.0-i/5.0
	ax.view_init(last_angle,angle)
	filename='3dplot/bench_step'+'x'*(count+i)+'.png'
	plt.savefig(filename)
count+=i
# slow endrange
for i, angle in enumerate(list(range(1,16,1))):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(pyg_x, pyg_y, pyg_z, label='pyg_lib fwd pass time')
	ax.scatter(vanilla_x, vanilla_y, vanilla_z, label='vanilla fwd pass time')
	ax.set_xlabel('# Node Types')
	ax.set_ylabel('# Edge Types')
	ax.set_zlabel('Forward Pass Time (s)')
	ax.legend()
	ax.view_init(last_angle, 355+angle/4.0)
	filename='3dplot/bench_step'+'x'*(count+i)+'.png'
	plt.savefig(filename)
count+=i
#goback
# slow endrange
for i, angle in enumerate(list(range(16,1,-1))):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(pyg_x, pyg_y, pyg_z, label='pyg_lib fwd pass time')
	ax.scatter(vanilla_x, vanilla_y, vanilla_z, label='vanilla fwd pass time')
	ax.set_xlabel('# Node Types')
	ax.set_ylabel('# Edge Types')
	ax.set_zlabel('Forward Pass Time (s)')
	ax.legend()
	ax.view_init(last_angle, 355+angle/4.0)
	filename='3dplot/bench_step'+'x'*(count+i)+'.png'
	plt.savefig(filename)
# fast back to start
count+=i
for i, angle in enumerate(range(355,270,-2)):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(pyg_x, pyg_y, pyg_z, label='pyg_lib fwd pass time')
	ax.scatter(vanilla_x, vanilla_y, vanilla_z, label='vanilla fwd pass time')
	ax.set_xlabel('# Node Types')
	ax.set_ylabel('# Edge Types')
	ax.set_zlabel('Forward Pass Time (s)')
	ax.legend()
	new_angle = last_angle + i/5.0
	ax.view_init(new_angle,angle)
	filename='3dplot/bench_step'+'x'*(count+i)+'.png'
	plt.savefig(filename)
#make gif:
#rm animated_bench.gif 3dplot/*; vim x; mv x x.py; python3 x.py; /opt/homebrew/Cellar/imagemagick/7.1.0-60/bin/convert -delay 10 3dplot/*.png animated_bench.gif
