import numpy as np
import matplotlib.pyplot as plt

occ_eucl = np.load('study/scores_occ_eucl100.npz')
tfidf_eucl = np.load('study/scores_tfidf_eucl100.npz')
tfidf_cosin= np.load('study/scores_tfidf_cosin100.npz')


data = [occ_eucl['R'], tfidf_eucl['R'], tfidf_cosin['R']]
# fig7, ax7 = plt.subplots()
# ax7.set_title('Multiple Samples with Different sizes')
plt.title('Recalls')
plt.xlabel('Modèles')
plt.boxplot(data)
plt.gca().get_xaxis().set_ticklabels( ['occ eucl', 'tfidf eucl', 'tfidf cosin'], fontsize=10)

plt.savefig('study/recalls.png')
plt.show()

data = [occ_eucl['P'], tfidf_eucl['P'], tfidf_cosin['P']]
# fig7, ax7 = plt.subplots()
# ax7.set_title('Multiple Samples with Different sizes')
plt.title('Precisions')
plt.xlabel('Modèles')
plt.boxplot(data)
plt.gca().get_xaxis().set_ticklabels( ['occ eucl', 'tfidf eucl', 'tfidf cosin'], fontsize=10)

plt.savefig('study/precisions.png')
plt.show()

data = [occ_eucl['F'], tfidf_eucl['F'], tfidf_cosin['F']]
# fig7, ax7 = plt.subplots()
# ax7.set_title('Multiple Samples with Different sizes')
plt.title('F scores')
plt.xlabel('Modèles')
plt.boxplot(data)
plt.gca().get_xaxis().set_ticklabels(['occ eucl', 'tfidf eucl', 'tfidf cosin'], fontsize=10)

plt.savefig('study/fscores.png')
plt.show()


tfidf_cosin['F'].mean()