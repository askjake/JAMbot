import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the JSON data
with open('network_data.json') as f:
    data = json.load(f)

# Create a graph object
G = nx.Graph()

# Iterate through the network data to add nodes and edges
for device_ip, details in data['devices'].items():
    if 'arp_entries' in details:
        # Add the main device as a node
        G.add_node(device_ip, label='Device', color='lightblue')

        # Add ARP entries as nodes and connect to the main device if they are not ghost devices
        for entry in details['arp_entries']:
            if not entry.get('ghost_mac', False):
                G.add_node(entry['ip'], label='ARP Device', color='lightgreen')
                G.add_edge(device_ip, entry['ip'])

# Add nodes for networks and connect devices to their respective networks
for network, devices in data.get('networks', {}).items():
    G.add_node(network, label='Network', color='lightyellow')
    for device in devices:
        G.add_edge(network, device)

# Extract node colors
colors = [G.nodes[n].get('color', 'lightgray') for n in G.nodes]

# Set structured positions in 3D for nodes
node_positions = {}
network_level = 0.0  # Z-axis level for networks
device_level = 1.0   # Z-axis level for devices
arp_level = -1.0     # Z-axis level for ARP entries

# Positioning the nodes based on their type
network_count = 0
device_count = 0
arp_count = 0

for node in G.nodes:
    if G.nodes[node]['label'] == 'Network':
        # Place networks in a row
        node_positions[node] = [network_count * 2.0, 0.0, network_level]
        network_count += 1
    elif G.nodes[node]['label'] == 'Device':
        # Place devices in a separate plane above the networks
        node_positions[node] = [device_count * 2.0, 5.0, device_level]
        device_count += 1
    elif G.nodes[node]['label'] == 'ARP Device':
        # Place ARP entries below the devices
        node_positions[node] = [arp_count * 2.0, -5.0, arp_level]
        arp_count += 1

# Prepare to plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the nodes
for node, position in node_positions.items():
    ax.scatter(position[0], position[1], position[2], color=G.nodes[node]['color'], s=100)

# Draw edges between nodes
for edge in G.edges:
    pos1 = node_positions[edge[0]]
    pos2 = node_positions[edge[1]]
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], color='gray', linewidth=0.5)

# Add labels to nodes
for node, position in node_positions.items():
    ax.text(position[0], position[1], position[2], node, fontsize=8)

# Set plot details
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('3D Network Vectograph (Structured Layout)')

# Show plot
plt.show()
