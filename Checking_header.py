import dpkt
import socket
import math

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
        ethernet_flow_buckets = {}
        raw_data_flow_buckets = {}

        for ts, buf in pcap:
            if idx == 0:
                first_ts = ts
                idx = 1
            timestamp = ts - first_ts

            # Check if the packet has an Ethernet header
            try:
                eth = dpkt.ethernet.Ethernet(buf)

                if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                    continue

                ip = eth.data
#                 if ip.p not in (dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP):
#                     continue

                if ip.p == dpkt.ip.IP_PROTO_TCP:
                    tcp = ip.data
                    connection_id = (ip.src, tcp.sport, ip.dst, tcp.dport)
                    reverse_connection_id = (ip.dst, tcp.dport, ip.src, tcp.sport)

                    flow = connection_id_to_str(connection_id)
                    reverse_flow = connection_id_to_str(reverse_connection_id)

                    if flow in ethernet_flow_buckets:
                        ethernet_flow_buckets[flow].append([timestamp, 1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0])
                    elif reverse_flow in ethernet_flow_buckets:
                        ethernet_flow_buckets[reverse_flow].append([timestamp, -1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0])
                    else:
                        ethernet_flow_buckets[flow] = [[timestamp, 1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0]]
                
                elif ip.p == dpkt.ip.IP_PROTO_UDP:
                    udp = ip.data
                    connection_id = (ip.src, udp.sport, ip.dst, udp.dport)
                    reverse_connection_id = (ip.dst, udp.dport, ip.src, udp.sport)

                    flow = connection_id_to_str(connection_id)
                    reverse_flow = connection_id_to_str(reverse_connection_id)

                    if flow in ethernet_flow_buckets:
                        ethernet_flow_buckets[flow].append([timestamp, 1, len(udp.data), 0])
                    elif reverse_flow in ethernet_flow_buckets:
                        ethernet_flow_buckets[reverse_flow].append([timestamp, -1, len(udp.data), 0])
                    else:
                        ethernet_flow_buckets[flow] = [[timestamp, 1, len(udp.data), 0]]
            
            except dpkt.UnpackError:
                pass

            # Check if the packet has a raw data header
            try:
                ip = dpkt.ip.IP(buf)

                if ip.p not in (dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP):
                    continue

                if ip.p == dpkt.ip.IP_PROTO_TCP:
                    tcp = ip.data
                    connection_id = (ip.src, tcp.sport, ip.dst, tcp.dport)
                    reverse_connection_id = (ip.dst, tcp.dport, ip.src, tcp.sport)

                    flow = connection_id_to_str(connection_id)
                    reverse_flow = connection_id_to_str(reverse_connection_id)

                    if flow in raw_data_flow_buckets:
                        raw_data_flow_buckets[flow].append([timestamp, 1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0])
                    elif reverse_flow in raw_data_flow_buckets:
                        raw_data_flow_buckets[reverse_flow].append([timestamp, -1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0])
                    else:
                        raw_data_flow_buckets[flow] = [[timestamp, 1, len(tcp.data), 1 if tcp.flags & dpkt.tcp.TH_FIN else 0]]
                
                elif ip.p == dpkt.ip.IP_PROTO_UDP:
                    udp = ip.data
                    connection_id = (ip.src, udp.sport, ip.dst, udp.dport)
                    reverse_connection_id = (ip.dst, udp.dport, ip.src, udp.sport)

                    flow = connection_id_to_str(connection_id)
                    reverse_flow = connection_id_to_str(reverse_connection_id)

                    if flow in raw_data_flow_buckets:
                        raw_data_flow_buckets[flow].append([timestamp, 1, len(udp.data), 0])
                    elif reverse_flow in raw_data_flow_buckets:
                        raw_data_flow_buckets[reverse_flow].append([timestamp, -1, len(udp.data), 0])
                    else:
                        raw_data_flow_buckets[flow] = [[timestamp, 1, len(udp.data), 0]]
            
            except dpkt.UnpackError:
                pass

        print("Ethernet Flow Buckets:")
        print(ethernet_flow_buckets)
        print('------------x---------------')
        print("Raw Data Flow Buckets:")
        print(raw_data_flow_buckets)

        ethernet_flow_buckets_count = len(ethernet_flow_buckets)
        raw_data_flow_buckets_count = len(raw_data_flow_buckets)

        print("Number of Ethernet Flow Buckets:", ethernet_flow_buckets_count)
        print("Number of Raw Data Flow Buckets:", raw_data_flow_buckets_count)

# Example usage
pcap_file = 'E:\PPT Reader\http_ipv4_complex.cap'
analyze_pcap_file(pcap_file)
