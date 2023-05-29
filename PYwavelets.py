#CSV File creation

import dpkt
import socket
import math
import statistics
import numpy as np
import pywt


def connection_id_to_str(cid, v=4):
    """This converts the connection ID cid which is a tuple of (source_ip_address, source_tcp_port, destination_ip_address,
    destination_tcp_port) to a string. v is either 4 for IPv4 or 6 for IPv6"""
    if v == 4:
        src_ip_addr_str = socket.inet_ntoa(cid[0])
        dst_ip_addr_str = socket.inet_ntoa(cid[2])
        return src_ip_addr_str + ":" + str(cid[1]) + "<=>" + dst_ip_addr_str + ":" + str(cid[3])
    elif v == 6:
        src_ip_addr_str = socket.inet_ntop(socket.AF_INET6, cid[0])
        dst_ip_addr_str = socket.inet_ntop(socket.AF_INET6, cid[2])
        return src_ip_addr_str + "." + str(cid[1]) + "<=>" + dst_ip_addr_str + "." + str(cid[3])
    else:
        raise ValueError('Argument to connection_id_to_str must be 4 or 6, is %d' % v)


def analyze_pcap_file(file_path):
    analyzed_data = []
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        idx = 0
        flow_buckets = {}
        total_duration = 0  # Initialize total duration
        for ts, buf in pcap:
            if idx == 0:
                first_ts = ts
                idx = 1
            timestamp = ts - first_ts
            eth = dpkt.ethernet.Ethernet(buf)
            if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                continue

            ip = eth.data
            if ip.p != dpkt.ip.IP_PROTO_TCP:
                continue

            total_duration = max(total_duration, timestamp)  # Update total duration

            tcp = ip.data
            connection_id = (ip.src, tcp.sport, ip.dst, tcp.dport)
            reverse_connection_id = (ip.dst, tcp.dport, ip.src, tcp.sport)

            flow = connection_id_to_str(connection_id)
            reverse_flow = connection_id_to_str(reverse_connection_id)

            if flow in flow_buckets:
                flow_buckets[flow].append([timestamp, 1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0])
            elif reverse_flow in flow_buckets:
                flow_buckets[reverse_flow].append([timestamp, -1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0])
            else:
                flow_buckets[flow] = [[timestamp, 1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0]]
                # Packet processing code

        No_of_windows = math.ceil(total_duration / window_size)
#         print("Flow Buckets:")
#         print(flow_buckets)
#         print('------------x---------------')
#         print("Total Duration:", total_duration)
#         print("Number of Windows:", No_of_windows)
        window_buckets = [{} for _ in range(No_of_windows)]
        
        for flow, packets in flow_buckets.items():
            
            for packet in packets:
               
                timestamp, direction, data_length, fin_flag = packet
                window_index = math.floor(timestamp / window_size)
                is_active=0
                active_start_time= -1
                buffer_time= 0
                if flow not in window_buckets[window_index]:
                    window_buckets[window_index][flow] = {
                        'total_forward_packets': 0,
                        'total_backward_packets': 0,
                        'total_bytes_per_sec': 0,
                        'forward_bytes': 0,
                        'backward_bytes': 0,
                        'forward_bins': [0] * (math.floor(window_size / bin_size) + change_bins_size),
                        'backward_bins': [0] * (math.floor(window_size / bin_size) + change_bins_size),
                        'forward_time':[],
                        'backward_time':[],
                        'forward_mean': 0,
                        'forward_min': 0,
                        'forward_max': 0,
                        'forward_std': 0,
                        'backward_mean': 0,
                        'backward_min': 0,
                        'backward_max': 0,
                        'backward_std': 0,
                        'flow_mean': 0,
                        'flow_min': 0,
                        'flow_max': 0,
                        'flow_std': 0,
                        'active': [],
                        'idle': [],
                        'buffer_time': 0,
                        'active_start_time': -1,
                        'is_active': 0,
                        'forward_relative_energy':[],
                        'backward_relative_energy':[],
                        'forward_shannon_entropy':[],
                        'backward_shannon_entropy':[],
                        'detail_forward_coffe':[],
                        'detail_backward_coffe':[],
                        'std_forward_coffe':[],
                        'std_backward_coffe':[]
                        
                    }
               

                flow_data = window_buckets[window_index][flow]
                
                if is_active == 0:
                    is_active = 1
                    active_start_time = timestamp
                    buffer_time = timestamp
                else:
                    if hard_timeout < (timestamp - buffer_time):
                        flow_data['active'].append(buffer_time - active_start_time)
                        flow_data['idle'].append(timestamp - buffer_time)
                    else:
                        buffer_time = timestamp

                bin_index = math.floor(timestamp % window_size / bin_size)
                if direction == 1:
                    flow_data['total_forward_packets'] += 1
                    flow_data['forward_bytes'] += data_length
                    flow_data['forward_bins'][bin_index] += data_length
                    flow_data['forward_time'].append(timestamp) 
                elif direction == -1:
                    flow_data['total_backward_packets'] += 1
                    flow_data['backward_bytes'] += data_length
                    flow_data['backward_bins'][bin_index] += data_length
                    flow_data['backward_time'].append(timestamp)   
                    
        for idx, window in enumerate(window_buckets):
            for flow, flow_data in window.items():
                forward_interarrival_time = []
                backward_interarrival_time = []
                flow_interarrival_time = []
#                
                forward_time = flow_data['forward_time']
                backward_time = flow_data['backward_time']

                for i in range(1, len(forward_time)):
                    forward_interarrival_time.append(forward_time[i] - forward_time[i - 1])

                for i in range(1, len(backward_time)):
                    backward_interarrival_time.append(backward_time[i] - backward_time[i - 1])
#                 print(forward_interarrival_time)
                merged_time = sorted(forward_time + backward_time)
                total_duration_of_flow = np.max(merged_time) - np.min(merged_time)
                total_bytes_per_sec = (flow_data['forward_bytes'] + flow_data['backward_bytes']) / total_duration_of_flow
                flow_data['total_bytes_per_sec'] = total_bytes_per_sec
                flow_interarrival_time.append([merged_time[i] - merged_time[i - 1] for i in range(1, len(merged_time))])

                if len(forward_interarrival_time) > 0:
                    forward_mean = np.mean(forward_interarrival_time)
                    forward_min = np.min(forward_interarrival_time)
                    forward_max = np.max(forward_interarrival_time)
                    forward_std = np.std(forward_interarrival_time)
                    flow_data['forward_mean'] = forward_mean
                    flow_data['forward_min'] = forward_min
                    flow_data['forward_max'] = forward_max
                    flow_data['forward_std'] = forward_std

                if len(backward_interarrival_time) > 0:
                    backward_mean = np.mean(backward_interarrival_time)
                    backward_min = np.min(backward_interarrival_time)
                    backward_max = np.max(backward_interarrival_time)
                    backward_std = np.std(backward_interarrival_time)
                    flow_data['backward_mean'] = backward_mean
                    flow_data['backward_min'] = backward_min
                    flow_data['backward_max'] = backward_max
                    flow_data['backward_std'] = backward_std

                if len(flow_interarrival_time) > 0:
                    flow_mean = np.mean(flow_interarrival_time)
                    flow_min = np.min(flow_interarrival_time)
                    flow_max = np.max(flow_interarrival_time)
                    flow_std = np.std(flow_interarrival_time)
                    flow_data['flow_mean'] = flow_mean
                    flow_data['flow_min'] = flow_min
                    flow_data['flow_max'] = flow_max
                    flow_data['flow_std'] = flow_std

                if hard_timeout < window_size * (idx + 1) - buffer_time:
                    flow_data['active'].append(buffer_time - active_start_time)
                    flow_data['idle'].append(window_size * (idx + 1) - buffer_time)
                else:
                    flow_data['active'].append(window_size * (idx + 1) - active_start_time)

                forward_bins = flow_data['forward_bins']
                backward_bins = flow_data['backward_bins']
                df = pywt.swt(forward_bins, 'haar', trim_approx=True)
                df1 = np.array(df)
                df_back = pywt.swt(backward_bins, 'haar', trim_approx=True)
                df3 = np.array(df_back)
                Shannon_entropy_forward=[]
                Shannon_entropy_back=[]
                # Calculate relative wavelet energy for each frequency band in forward bins
                for k in range(len(df1)):
                    energy_kf = np.sum(df1[k])**2
                    frequency_energy_forward=np.sum(energy_kf)
                    relative_energy_k = energy_kf / frequency_energy_forward
                    flow_data['forward_relative_energy'].append(relative_energy_k)
                    
                    #shannon entropy
                    shannon_entropy_dnk_forward = np.abs(df1[k])**2
                    fraction_of_energy_forward = (shannon_entropy_dnk_forward / frequency_energy_forward)
                    Shan_Entropy_log= -np.sum(fraction_of_energy_forward * np.log(fraction_of_energy_forward))
                    flow_data['forward_shannon_entropy'].append(Shan_Entropy_log)
                    
                    #Detail Coefficient Statistics
                    
                    detail_energy_forward = np.sum(df1[k])
                    Detail_Coefficient_forward = detail_energy_forward / len(df1)
                    flow_data['detail_forward_coffe'].append(Detail_Coefficient_forward)
                    
                    #standard deviation of the detail coefficients_forward
                    
                    std_forward_coffe= np.sqrt((np.sum(Detail_Coefficient_forward - (df1[k]))) / len(df1))
                    flow_data['std_forward_coffe'].append(std_forward_coffe)
                    
                    
                    
                # Calculate relative wavelet energy for each frequency band in backward bins
                for k in range(len(df3)):
                    energy_kb = np.sum(df3[k])**2
                    frequency_energy_backward=np.sum(energy_kb)
                    relative_energy_k = energy_kb /frequency_energy_backward
                    flow_data['backward_relative_energy'].append(relative_energy_k)
                    
                    #shannon entropy
                    
                    shannon_entropy_dnk_backward = np.abs(df3[k])**2
                    fraction_of_energy_backward = (shannon_entropy_dnk_backward / frequency_energy_backward)
                    Shan_Entropy_log= -np.sum(fraction_of_energy_backward * np.log(fraction_of_energy_backward))
                    flow_data['backward_shannon_entropy'].append(Shan_Entropy_log)
    
                    #Detail Coefficient Statistics
                    
                    detail_energy_backward = np.sum(df3[k])
                    Detail_Coefficient_backward = detail_energy_backward / len(df3)
                    flow_data['detail_backward_coffe'].append(Detail_Coefficient_backward)
                    
                    #standard deviation of the detail coefficients_backward
                    
                    
                    std_back_coffe= np.sqrt((np.sum(Detail_Coefficient_backward - (df3[k]))) / len(df3))
                    flow_data['std_backward_coffe'].append(std_back_coffe)
               
                    
        for idx, window in enumerate(window_buckets):
            print("Window", idx)
            for key, value in window.items():
                print(key)
                print(value) 
    return analyzed_data

def write_to_csv(data, file_path):
    with open(file_path, 'w', newline='|') as f:
        writer = csv.writer(f)
        writer.writerow([
            'total_forward_packets',
            'Flow',
            'Total Forward Packets',
            'Total Backward Packets',
            'Total Bytes per Second',
            'Flow Mean',
            'Flow Min',
            'Flow Max',
            'Flow Standard Deviation',
            'Flow Relative Energy',
            'Flow Shannon Entropy',
            'Flow Detail Coefficient',
            'Flow Standard Coefficient'
        ])
        writer.writerows(data)


file_path = 'E:\PPT Reader\http_ipv4_complex.cap'

analyzed_data = analyze_pcap_file(file_path)
write_to_csv(analyzed_data, 'E:/analyzed_data.csv')
print('Analysis completed. Data written to analyzed_data.csv.')
# change_bins_size = 12
hard_timeout = 0.1
window_size = 2
bin_size = 0.1
