import matplotlib.pyplot as plt
pyg_lib_times = {(2, 4): 8.069992065429688e-05, (2, 8): 9.047508239746094e-05, (2, 16): 0.00011226177215576172, (2, 32): 0.0001558685302734375, (2, 64): 0.0002344369888305664, (2, 128): 0.00042046546936035155, (2, 256): 0.000760807991027832, (4, 8): 9.900569915771484e-05, (4, 16): 0.00011890888214111328, (4, 32): 0.00015993118286132811, (4, 64): 0.00024888992309570314, (4, 128): 0.00043740272521972654, (4, 256): 0.000765223503112793, (8, 16): 0.00013247489929199218, (8, 32): 0.00017428874969482421, (8, 64): 0.00026771068572998044, (8, 128): 0.00044491291046142576, (8, 256): 0.0007893514633178711, (16, 32): 0.00019882678985595704, (16, 64): 0.0002849912643432617, (16, 128): 0.0004630327224731445, (16, 256): 0.0008281087875366211, (32, 64): 0.00033455848693847656, (32, 128): 0.0005031728744506836, (32, 256): 0.0008721208572387695, (64, 128): 0.0006217145919799804, (64, 256): 0.0009572410583496094, (128, 256): 0.0011648941040039063}
vanilla_times = {(2, 4): 5.157470703125e-05, (2, 8): 9.539127349853516e-05, (2, 16): 0.00018222808837890626, (2, 32): 0.00035670280456542966, (2, 64): 0.0007035255432128907, (2, 128): 0.001425161361694336, (2, 256): 0.0028867721557617188, (4, 8): 0.00010233402252197266, (4, 16): 0.00019299983978271484, (4, 32): 0.000367889404296875, (4, 64): 0.0007239055633544922, (4, 128): 0.0014357185363769532, (4, 256): 0.0028855085372924806, (8, 16): 0.00020418643951416015, (8, 32): 0.00038424015045166013, (8, 64): 0.0007417964935302735, (8, 128): 0.0014571523666381835, (8, 256): 0.0029382848739624022, (16, 32): 0.0004047250747680664, (16, 64): 0.0007723474502563477, (16, 128): 0.0015054941177368164, (16, 256): 0.004117259979248047, (32, 64): 0.0008162641525268554, (32, 128): 0.0015496587753295898, (32, 256): 0.003016548156738281, (64, 128): 0.0016500186920166016, (64, 256): 0.0031356239318847657, (128, 256): 0.0033610916137695314}
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
