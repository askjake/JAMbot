import subprocess
import sys
import threading
import json
import os
import socket
from scapy.all import *
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# File to store the discovered network data
DATA_FILE = 'network_data.json'
GHOST_MACS = ["00:00:0c:9f:f1:7e"]
MAX_IP_THRESHOLD = 10

# Function to perform traceroute and get the list of hops
def perform_traceroute(destination_ip, max_hops=5):
    print(f"Performing traceroute to {destination_ip} with max {max_hops} hops...")
    hops = []
    for ttl in range(1, max_hops + 1):
        pkt = IP(dst=destination_ip, ttl=ttl) / ICMP()
        reply = sr1(pkt, verbose=0, timeout=2)
        if reply is None:
            break
        else:
            hops.append(reply.src)
            if reply.src == destination_ip:
                break
    return hops

# Function to get ARP table entries from a device
def get_arp_table(ip_address):
    print(f"Retrieving ARP table from {ip_address}...")
    arp_entries = []
    ans, unans = sr(ARP(pdst=f"{ip_address}/24"), timeout=2, verbose=0)
    for snd, rcv in ans:
        is_ghost = rcv.hwsrc in GHOST_MACS
        arp_entries.append({
            'ip': rcv.psrc,
            'mac': rcv.hwsrc,
            'ghost_mac': is_ghost
        })
    return arp_entries

# Load or initialize the network data file
def load_network_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {'devices': {}, 'networks': {}, 'connections': []}

# Save network data to a file
def save_network_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Function to identify ghost MACs based on IP assignments
def identify_ghost_macs(network_data):
    mac_ip_count = {}

    # Count IP addresses per MAC
    for device, details in network_data['devices'].items():
        mac = details.get('mac')
        if mac:
            if mac not in mac_ip_count:
                mac_ip_count[mac] = set()
            mac_ip_count[mac].add(device)

    # Identify ghost MACs based on the threshold
    for mac, ips in mac_ip_count.items():
        if len(ips) > MAX_IP_THRESHOLD:
            print(f"Flagging {mac} as a ghost MAC due to {len(ips)} IP addresses.")
            GHOST_MACS.append(mac)

# Function to build or update the network map
def update_network_map(hops, network_data):
    new_data_found = False
    for hop in hops:
        if hop not in network_data['devices']:
            arp_entries = get_arp_table(hop)
            # Filter out ghost MAC addresses
            arp_entries = [entry for entry in arp_entries if not entry['ghost_mac']]
            
            network_data['devices'][hop] = {
                'arp_entries': arp_entries,
                'last_checked': time.time()
            }
            new_data_found = True

            # Add the device to its respective network
            network_prefix = '.'.join(hop.split('.')[:3])
            network = f"{network_prefix}.0/24"
            if network not in network_data['networks']:
                network_data['networks'][network] = []
            if hop not in network_data['networks'][network]:
                network_data['networks'][network].append(hop)

            # Add connections between the device and its ARP entries
            for entry in arp_entries:
                if entry['ip'] not in network_data['devices']:
                    network_data['devices'][entry['ip']] = {
                        'mac': entry['mac'],
                        'last_checked': time.time()
                    }
                if {'from': hop, 'to': entry['ip']} not in network_data['connections']:
                    network_data['connections'].append({'from': hop, 'to': entry['ip']})

    # Identify and flag ghost MACs after the map is updated
    identify_ghost_macs(network_data)
    
    return new_data_found

# Function to display the network map graphically
def display_network_map(network_data, target_network):
    G = nx.Graph()

    # Add nodes for devices and networks, with the target network highlighted
    for network, devices in network_data['networks'].items():
        if network == target_network:
            G.add_node(network, label=network, color='green', shape='cloud')
        else:
            G.add_node(network, label=network, color='lightgreen', shape='cloud')

        for device in devices:
            G.add_node(device, label=device, color='lightblue', shape='circle')
            G.add_edge(network, device)

    # Add edges for direct connections between devices
    for connection in network_data['connections']:
        G.add_edge(connection['from'], connection['to'])

    # Create interactive plot with Tkinter
    root = tk.Tk()
    root.title(f"Network Map for {target_network}")

    # Get and display the computer's hostname
    hostname = socket.gethostname()
    tk.Label(root, text=f"Computer Name: {hostname}", font=("Arial", 14)).pack(pady=10)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, 
        with_labels=True, 
        labels={n: n for n in G.nodes}, 
        node_color=[G.nodes[n].get('color', 'lightblue') for n in G.nodes], 
        edge_color='gray', 
        node_size=1500, 
        ax=ax
    )
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def expand_node(node):
        print(f"Expanding node {node}...")
        if node in network_data['devices']:
            arp_entries = get_arp_table(node)
            # Filter out ghost MAC addresses
            arp_entries = [entry for entry in arp_entries if not entry['ghost_mac']]
            
            network_data['devices'][node]['arp_entries'] = arp_entries
            update_network_map([node], network_data)
            save_network_data(network_data)
            # Redraw the updated graph
            ax.clear()
            nx.draw(
                G, pos, 
                with_labels=True, 
                labels={n: n for n in G.nodes}, 
                node_color=[G.nodes[n].get('color', 'lightblue') for n in G.nodes], 
                edge_color='gray', 
                node_size=1500, 
                ax=ax
            )
            canvas.draw()

    def on_click(event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            closest_node = min(G.nodes, key=lambda n: (pos[n][0] - x) ** 2 + (pos[n][1] - y) ** 2)
            print(f"Clicked on node {closest_node}")
            expand_node(closest_node)

    fig.canvas.mpl_connect('button_press_event', on_click)
    tk.mainloop()

def main():
    destination_ip = input("Enter the destination IP address to trace: ")
    max_hops = int(input("Enter the maximum number of hops: "))

    # Load previously stored network data
    network_data = load_network_data()

    # Perform traceroute and update the network map
    hops = perform_traceroute(destination_ip, max_hops)
    print(f"Discovered hops: {hops}")

    # Update the network map with newly discovered hops
    if update_network_map(hops, network_data):
        save_network_data(network_data)

    # Determine the target network for the destination IP
    network_prefix = '.'.join(destination_ip.split('.')[:3])
    target_network = f"{network_prefix}.0/24"

    print("Network map updated.")
    display_network_map(network_data, target_network)

if __name__ == "__main__":
    main()
