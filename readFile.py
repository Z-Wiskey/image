import numpy as np

# emb = np.load('emb.npy', allow_pickle=True)
#
# print(emb)

mobility = np.load('./similarity/similarity_Mobility_vsgre.npy', allow_pickle=True)

print(mobility)

poi = np.load('./similarity/similarity_POI_vsgre.npy', allow_pickle=True)

print(poi)

si_embedding = np.load('./prediction-tasks/emb/si_embedding.npy', allow_pickle=True)

print(si_embedding)

sv_embedding = np.load('./prediction-tasks/emb/sv_embedding.npy', allow_pickle=True)

print(sv_embedding)






