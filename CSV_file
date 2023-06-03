import dpkt
import socket
import math
import statistics
import numpy as np
import pywt
import csv
import pandas as pd


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
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        idx = 0
        flow_buckets = {}
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
            
            total_duration=ts-first_ts
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
        print("Flow Buckets:")
        print(flow_buckets)
        print('------------x---------------')
        print("Total Duration:", total_duration)
        print("Number of Windows:", No_of_windows)
        window_buckets = [{} for _ in range(No_of_windows)]
        
        for flow, packets in flow_buckets.items():
            for packet in packets:
                timestamp, direction, data_length, fin_flag = packet
                window_index = math.floor(timestamp / window_size)
                if flow not in window_buckets[window_index]:
                    window_buckets[window_index][flow] = {
                        'total_forward_packets': 0,
                        'total_backward_packets': 0,'total_bytes_per_sec': 0,
                        'forward_bytes': 0,
                        'backward_bytes': 0,
                        'forward_time':[],
                        'backward_time':[],
                        'forward_bins':[0]* (math.floor(window_size / bin_size) + 12),
                        'backward_bins':[0]* (math.floor(window_size / bin_size) + 12),
                        'forward_mean': 0,'forward_min': 0, 'forward_max': 0,'forward_std': 0, 
                        'backward_mean': 0,  'backward_min': 0,'backward_max': 0,'backward_std': 0,
                        'flow_mean': 0,'flow_min': 0, 'flow_max': 0, 'flow_std': 0,
                        'active_mean':0, 'active_min':0, 'active_max':0, 'active_std':0,
                        'idle_mean':0, 'idle_min':0,'idle_max':0, 'idle_std':0,
                        'buffer_time':0,'active_start_time':-1,'is_active':0,
                        'forward_relative_energy':[], 'backward_relative_energy':[],
                        'forward_shannon_entropy':[],'backward_shannon_entropy':[],
                        'detial_forward_coffe':[], 'detial_backward_coffe':[],
                        'std_forward_coffe':[], 'std_backward_coffe':[]
                    }
                
                flow_data = window_buckets[window_index][flow]
                active=[]
                idle=[]
                if flow_data['is_active']==0:
                    
                    flow_data['is_active']=1
                    flow_data['active_start_time']=timestamp
                    flow_data['buffer_time']=timestamp
                else:
                    if hard_timeout < (timestamp-flow_data['buffer_time']):
                        
                        active.append(flow_data['buffer_time']- flow_data['active_start_time'])
                        idle.append(timestamp-flow_data['buffer_time'])
                    else:
                        
                        flow_data['buffer_time']=timestamp  
               
            
                bin_index = math.floor(timestamp%window_size / bin_size)
                if direction == 1:
                    flow_data['total_forward_packets'] += 1
                    flow_data['forward_bytes'] += data_length
                    flow_data['forward_time'].append(timestamp)
                    flow_data['forward_bins'][bin_index] += data_length
                elif direction == -1:
                    flow_data['total_backward_packets'] += 1
                    flow_data['backward_bytes'] += data_length
                    flow_data['backward_time'].append(timestamp)
                          
                    flow_data['backward_bins'][bin_index] += data_length
        
        remove_keys=[]
        for idx,window in enumerate(window_buckets):
            total_packets=sum(flow_data['total_forward_packets']+ flow_data['total_backward_packets'] for flow_data in window.values())
            if total_packets < Minimum_Window_packet_size:
                print("Window",idx)
                remove_keys.append(idx)
                continue
            for flow,flow_data in window.items():
                forward_time=flow_data['forward_time']
                backward_time=flow_data['backward_time']
                forward_interarrival_time=[]
                backward_interarrival_time=[]
                flow_interarrival_time=[]
                for i in range(1, len(forward_time)):
                    forward_interarrival_time.append(forward_time[i] - forward_time[i-1])

                for i in range(1, len(backward_time)):
                    backward_interarrival_time.append(backward_time[i] - backward_time[i-1])
            
                merged_time = sorted(forward_time + backward_time)
                total_duration_of_flow= np.max(merged_time)-np.min(merged_time)
                total_bytes_per_sec = (flow_data['forward_bytes'] + flow_data['backward_bytes']) / total_duration_of_flow
                flow_data['total_bytes_per_sec'] = total_bytes_per_sec
                flow_interarrival_time.append([merged_time[i] - merged_time[i - 1] for i in range(1, len(merged_time))])
        
                if len(forward_interarrival_time) > 0:
        
                    flow_data['forward_mean'] = np.mean(forward_interarrival_time)
                    flow_data['forward_min']= np.min(forward_interarrival_time)
                    flow_data['forward_max'] = np.max(forward_interarrival_time)
                    flow_data['forward_std'] = np.std(forward_interarrival_time)
                   
                if len(backward_interarrival_time) > 0:
                    flow_data['backward_mean'] = np.mean(backward_interarrival_time)
                    flow_data['backward_min'] = np.min(backward_interarrival_time)
                    flow_data['backward_max'] = np.max(backward_interarrival_time)
                    flow_data['backward_std']= np.std(backward_interarrival_time)
                    
                if len(flow_interarrival_time) > 0:
                    
                    flow_data['flow_mean'] = np.mean(flow_interarrival_time)
                    flow_data['flow_min'] = np.min(flow_interarrival_time)
                    flow_data['flow_max'] = np.max(flow_interarrival_time)
                    flow_data['flow_std'] = np.std(flow_interarrival_time)
                
                
                if hard_timeout < window_size*(idx+1) -flow_data['buffer_time']:
                    active.append(flow_data['buffer_time'] - flow_data['active_start_time'])
                    idle.append(window_size * (idx+1)- flow_data['buffer_time'])
                else:
                    active.append(window_size * (idx+1)-flow_data['active_start_time'])
                    
                
                if len(active) > 0:
                    flow_data['active_mean'] = np.mean(active)
                    flow_data['active_min'] = np.min(active)
                    flow_data['active_max'] = np.max(active)
                    flow_data['active_std']= np.std(active)
                
                if len(idle) > 0:
                    flow_data['idle_mean'] = np.mean(idle)
                    flow_data['idle_min'] = np.min(idle)
                    flow_data['idle_max'] = np.max(idle)
                    flow_data['idle_std']= np.std(idle)
                
                
                forward_bins = flow_data['forward_bins']
                backward_bins = flow_data['backward_bins']
                df = pywt.swt(forward_bins, 'haar', trim_approx=True)
                df1 = np.array(df)
                df2 = pywt.swt(forward_bins, 'haar', trim_approx=True)
                df3 = np.array(df)
                
                energy_frequency=[]
                Ek_freq=[]
                rows,col=(6,32)
                arr_fraction_rho=[[0] * col] *rows
                log_function=[]
                detail_energy_forward=[]
                energy_frequency_without_abs=[]
                mu_k_list_forward=[]
                
                for k in range (len(df1)):
                    energy_frequency.append(np.abs(df1[k]**2))
                    energy_frequency_without_abs.append(df1[k]**2)
                    Ek_freq.append(np.sum(energy_frequency))
                    detail_energy_forward.append((np.sum(np.abs(df1[k])))/len(df1[k]))
                E_total_forward=np.sum(Ek_freq)   
                
                for i in range (len(Ek_freq)):
                    if E_total_forward !=0:
                        flow_data['forward_relative_energy'].append(Ek_freq[i]/E_total_forward)
                     
                    else:
                        flow_data['forward_relative_energy'].append(0)
                        
                    for j in range(len(energy_frequency_without_abs)):
                        if Ek_freq[i] !=0:
                            arr_fraction_rho[i][j]=(energy_frequency_without_abs[i][j]/Ek_freq[i])
                        else:
                            arr_fraction_rho[i][j]!=0
                 
                    if arr_fraction_rho[i][j] !=0:
                        flow_data['forward_shannon_entropy'].append(-np.sum(arr_fraction_rho[i][j] * np.log(arr_fraction_rho[i][j])))

                    else:
                        flow_data['forward_shannon_entropy'].append(0)
                 
                
                for i in range(len(detail_energy_forward)):
                    mu_k_list_forward.append(detail_energy_forward[i])
                
                flow_data['detial_forward_coffe']=mu_k_list_forward
                first_differne_step_for=[]
                for i in range(len(mu_k_list_forward)):
                    for j in range(len(df1[i])):
                        
                        first_differne_step_for.append((mu_k_list_forward[i] - df1[i][j])**2)
                    flow_data['std_forward_coffe'].append(np.sqrt(np.sum(first_differne_step_for)) / len(df1[i]))
               
                energy_frequency_back=[]
                Ek_freq_back=[]
                rows,col=(6,32)
                arr_fraction_rho_back=[[0] * col] *rows
                log_function_back=[]
                detail_energy_backward=[]
                energy_frequency_without_abs_back=[]
                mu_k_list_backward=[]
                
                for k in range (len(df3)):
                    energy_frequency_back.append(np.abs(df3[k]**2))
                    energy_frequency_without_abs_back.append(df3[k]**2)
                    Ek_freq_back.append(np.sum(energy_frequency_back))
                    detail_energy_backward.append((np.sum(np.abs(df3[k])))/len(df3[k]))
                E_total_backward=np.sum(Ek_freq_back)   
                
                for i in range (len(Ek_freq_back)):
                    if E_total_backward !=0:
                        flow_data['backward_relative_energy'].append(Ek_freq_back[i]/E_total_backward)
                     
                    else:
                        flow_data['backward_relative_energy'].append(0)
                        
                    for j in range(len(energy_frequency_without_abs)):
                        if Ek_freq_back[i] !=0:
                            arr_fraction_rho_back[i][j]=(energy_frequency_without_abs_back[i][j]/Ek_freq_back[i])
                        else:
                            arr_fraction_rho_back[i][j]!=0
                 
                    if arr_fraction_rho_back[i][j] !=0:
                        flow_data['backward_shannon_entropy'].append(-np.sum(arr_fraction_rho_back[i][j] * np.log(arr_fraction_rho_back[i][j])))

                    else:
                        flow_data['backward_shannon_entropy'].append(0)
                 
                
                for i in range(len(detail_energy_backward)):
                    mu_k_list_backward.append(detail_energy_backward[i])
                
                flow_data['detial_backward_coffe']=mu_k_list_backward
                first_differne_step_back=[]
                for i in range(len(mu_k_list_backward)):
                    for j in range(len(df3[i])):
                        
                        first_differne_step_back.append((mu_k_list_backward[i] - df3[i][j])**2)
                    flow_data['std_backward_coffe'].append(np.sqrt(np.sum(first_differne_step_back)) / len(df3[i]))
                    
            for value in list(reversed(remove_keys)):
                del window_buckets[value]
        
            print("Window",idx)
            print("---------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>YYYYYYYYYYYYYYYYYYYY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>------------")
            for key, value in window.items():
                print(key)
                print(value)
                print("----------------------------------------------------X---------------------------------------------")
        csv_file = 'E:/pcap_analysis.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Window', 'Flow', 'Total Forward Packets', 'Total Backward Packets', 'Total Bytes Per Second',
                             'Forward Bytes', 'Backward Bytes', 'Forward Mean', 'Forward Min', 'Forward Max',
                             'Forward Std', 'Backward Mean', 'Backward Min', 'Backward Max', 'Backward Std',
                             'Flow Mean', 'Flow Min', 'Flow Max', 'Flow Std', 'Active Mean', 'Active Min', 'Active Max',
                             'Active Std', 'Idle Mean', 'Idle Min', 'Idle Max', 'Idle Std', 'Forward Relative Energy',
                             'Backward Relative Energy', 'Forward Shannon Entropy', 'Backward Shannon Entropy',
                             'Detail Forward Coffe', 'Detail Backward Coffe', 'Std Forward Coffe', 'Std Backward Coffe'])

            for window_index, window_data in enumerate(window_buckets):
                for flow, flow_data in window_data.items():
                    writer.writerow([window_index, flow, flow_data['total_forward_packets'],
                                     flow_data['total_backward_packets'], flow_data['total_bytes_per_sec'],
                                     flow_data['forward_bytes'], flow_data['backward_bytes'],
                                     flow_data['forward_mean'], flow_data['forward_min'], flow_data['forward_max'],
                                     flow_data['forward_std'], flow_data['backward_mean'], flow_data['backward_min'],
                                     flow_data['backward_max'], flow_data['backward_std'], flow_data['flow_mean'],
                                     flow_data['flow_min'], flow_data['flow_max'], flow_data['flow_std'],
                                     flow_data['active_mean'], flow_data['active_min'], flow_data['active_max'],
                                     flow_data['active_std'], flow_data['idle_mean'], flow_data['idle_min'],
                                     flow_data['idle_max'], flow_data['idle_std'], flow_data['forward_relative_energy'],
                                     flow_data['backward_relative_energy'], flow_data['forward_shannon_entropy'],
                                     flow_data['backward_shannon_entropy'], flow_data['detial_forward_coffe'],
                                     flow_data['detial_backward_coffe'], flow_data['std_forward_coffe'],
                                     flow_data['std_backward_coffe']])

        print("Analysis complete. Data written to", csv_file)
       
                
hard_timeout = 0.1
Minimum_Window_packet_size = 2
bin_size = 0.1
analyze_pcap_file('E:\PPT Reader\http_ipv4_complex.cap')   
