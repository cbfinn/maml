
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# data from MAML
fc1 = [3961.19, 3977.48, 3937.99]
fc2 = [1151.23, 1302.56, 1449.12]
fc3 = [954.14, 1058.83, 978.62]
fc4 = [839.88, 879.96, 859.62]
fc5 = [839.82, 838.84, 796.00]

# data from MAML, with x in (-3,3), coeffs in (-1,1)
fc1 = [4.232, 4.523, 5.477]
fc2 = [3.797, 3.308, 3.152]
fc3 = [3.507, 2.982, 2.766]
fc4 = [2.163, 3.453, 2.552]
fc5 = [2.228, 2.238, 1.937]
maml_means = [np.mean(fc1), np.mean(fc2), np.mean(fc3), np.mean(fc4), np.mean(fc5)]
maml_stds = [np.std(fc1), np.std(fc2), np.std(fc3), np.std(fc4), np.std(fc5)]

# data from MAML with number of parameters fixed
#fc1 = []
fc1 = [4.147, 4.154, 3.671]  # with 250 per hidden layer instead of 100: 4.147, 4.154, 3.671
fc2 = [3.516, 2.939, 4.492]
fc3 = [2.640, 2.615, 2.756]
fc4 = [2.163, 2.479, 2.663]
fc5 = [2.228, 2.238, 1.937]
maml_fix_means = [np.mean(fc1), np.mean(fc2), np.mean(fc3), np.mean(fc4), np.mean(fc5)]
maml_fix_stds = [np.std(fc1), np.std(fc2), np.std(fc3), np.std(fc4), np.std(fc5)]

# data from oracle
oracle_fc1 = [3.602, 4.358, 4.560]
oracle_fc2 = [4.605, 5.916, 4.198]
oracle_fc3 = [4.310, 2.249, 4.764]

# data from oracle with x in (-3, 3), coeffs in (-1,1)
oracle_fc1 = [5.55e-3, 5.09e-3, 5.03e-3, 5.59e-3, 5.19e-3, 5.03e-3]
oracle_fc2 = [3.72e-3, 4.18e-3, 8.30e-3, 4.92e-2, 4.89e-3, 6.84e-3]
oracle_fc3 = [2.04e-3, 1.11e-2, 3.07e-3, 2.14e-3, 3.85e-3, 4.86e-3]
oracle_fc4 = [2.35e-3, 1.07e-3, 1.22e-3, 6.98e-2, 3.07e-3, 3.94e-3]
oracle_fc5 = [3.48e-3, 1.11e-3, 1.13e-3, 7.22e-2, 1.52e-3, 1.53e-3]
oracle_means = [np.mean(oracle_fc1), np.mean(oracle_fc2), np.mean(oracle_fc3), np.mean(oracle_fc4), np.mean(oracle_fc5)]
oracle_stds = [np.std(oracle_fc1), np.std(oracle_fc2), np.std(oracle_fc3), np.std(oracle_fc4), np.std(oracle_fc5)]

# data from MAML with number of parameters fixed
oracle_fc1 = [2.7e-3, 6.3e-3, 2.5e-3] # with 250 hidden units
oracle_fc2 = [9.1e-3, 2.1e-3, 2.1e-3]
oracle_fc3 = [6.0e-3, 9.4e-3, 2.3e-3]
oracle_fc4 = [8.2e-4, 3.2e-2, 3.0e-3]
oracle_fc5 = [3.48e-3, 1.11e-3, 1.13e-3, 7.22e-2, 1.52e-3, 1.53e-3]
oracle_fix_means = [np.mean(oracle_fc1), np.mean(oracle_fc2), np.mean(oracle_fc3), np.mean(oracle_fc4), np.mean(oracle_fc5)]
oracle_fix_stds = [np.std(oracle_fc1), np.std(oracle_fc2), np.std(oracle_fc3), np.std(oracle_fc4), np.std(oracle_fc5)]



plt.figure(figsize=(2*5.8, 2*4.8))
font_size = 38
tick_size = 27

plt.bar(np.arange(5)+1.0, maml_means, 0.35, color=["#8259b1","#75aa56","#b94c75","#66a1e5","#be7239"], yerr=maml_stds)
#plt.yscale('log')
plt.title('MAML 40-shot regression, 100 units per layer')
plt.xlabel('number of hidden layers')
plt.ylabel('40-shot mean-squared error')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('/home/cfinn/maml_depth.png')



plt.clf()
plt.bar(np.arange(5)+1.0, maml_fix_means, 0.35, color=["#8259b1","#75aa56","#b94c75","#66a1e5","#be7239"], yerr=maml_fix_stds, error_kw={'elinewidth':5})
plt.title('MAML, fixed parameter count', fontsize=font_size)
plt.xlabel('number of hidden layers', fontsize=font_size)
plt.ylabel('40-shot mean-squared error', fontsize=font_size)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=tick_size)
plt.savefig('/home/cfinn/maml_fix_depth.png')


plt.clf()
plt.bar(np.arange(5)+1.0, oracle_means, 0.35, color=["#8259b1","#75aa56","#b94c75"], yerr=oracle_stds)
plt.title('Task-conditioned, 100 units per layer', fontsize=font_size)
plt.xlabel('number of hidden layers', fontsize=font_size)
plt.ylabel('mean-squared error')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.ylim([0, 0.05])
plt.ylim([0, 0.1])
plt.xticks([1,2,3, 4, 5])
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('/home/cfinn/oracle_depth.png')


plt.clf()
plt.bar(np.arange(5)+1.0, oracle_fix_means, 0.35, color=["#8259b1","#75aa56","#b94c75","#66a1e5","#be7239"], yerr=oracle_fix_stds, error_kw={'elinewidth':5})
#plt.title('Oracle performance vs. depth, 40k total parameters')
plt.title('Task-conditioned, fixed parameter count', fontsize=font_size)  # 40k parameters
plt.xlabel('number of hidden layers', fontsize=font_size)
plt.ylim([0, 0.095])
plt.ylabel('mean-squared error', fontsize=font_size)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=tick_size)
plt.savefig('/home/cfinn/oracle_fix_depth.png')



