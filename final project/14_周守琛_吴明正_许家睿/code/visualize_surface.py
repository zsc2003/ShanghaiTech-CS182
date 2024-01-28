import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# path = '/4Tdisk/protein_surface/output/4F9A_AB_patch_0_ADP.npz'

path = '/4Tdisk/protein_surface/output/gdb_002627.npz'
path = '/4Tdisk/protein_surface/output/gdb_133881.npz'

pkt_mesh = np.load(path, allow_pickle=True)
# print(pkt_mesh['label'])
# print(pkt_mesh['atom_info'].shape)
# print(pkt_mesh['pkt_verts'].shape)
# print(pkt_mesh['pkt_faces'].shape)
print(pkt_mesh['eigen_vals'].shape)
print(pkt_mesh['eigen_vecs'].shape)
# print(pkt_mesh['mass'])

# seletect_id = 269
seletect_id = 0


intensity = np.einsum("i,bi->b", pkt_mesh['eigen_vals'][seletect_id:], pkt_mesh['eigen_vecs'][:, seletect_id:])
# colormap = plt.cm.get_cmap('PiYG')
colormap = plt.cm.get_cmap('twilight_shifted')
normalized_values = (intensity - intensity.min()) / (intensity.max() - intensity.min())
colors = colormap(normalized_values)[:, :3]
print(colors.shape)

# select part of the points of inner surface, the color of outer surface is is the mean of inner surface
# regard the inner points has continuous index
# inner_index = [i for i in range(100,903)]
# outer_index = [i for i in range(seletect_id) if i not in inner_index]

# # outer surface is white
# colors[outer_index] = [1,1,1]
# np.mean(colors[inner_index], axis=0)





mesh = o3d.geometry.TriangleMesh()

# mesh.vertices = o3d.utility.Vector3dVector(pkt_mesh['pkt_verts'])
# mesh.triangles = o3d.utility.Vector3iVector(pkt_mesh['pkt_faces'])
mesh.vertices = o3d.utility.Vector3dVector(pkt_mesh['verts'])
mesh.triangles = o3d.utility.Vector3iVector(pkt_mesh['faces'])

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])


