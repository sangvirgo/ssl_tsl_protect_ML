import pyshark
import pandas as pd
from collections import defaultdict
import numpy as np
from datetime import datetime

# Định nghĩa danh sách các đặc trưng cần lấy
network_features = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Packet Length Mean', 'Packet Length Std', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Timestamp', 'Label'
]

# Hàm tính toán các đặc trưng từ file .pcapng
def extract_features(pcap_file):
    # Khởi tạo dictionary để lưu trữ các luồng (flows)
    flows = defaultdict(lambda: {
        'Total Fwd Packets': 0, 'Total Backward Packets': 0,
        'Total Length of Fwd Packets': 0, 'Total Length of Bwd Packets': 0,
        'Fwd Packet Length Max': 0, 'Fwd Packet Length Min': float('inf'),
        'Bwd Packet Length Max': 0, 'Bwd Packet Length Min': float('inf'),
        'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
        'PSH Flag Count': 0, 'ACK Flag Count': 0,
        'Timestamp': '', 'Label': 'BENIGN'
    })

    try:
        cap = pyshark.FileCapture(pcap_file, display_filter='tcp || udp')
    except Exception as e:
        print(f"Error opening pcap file: {e}")
        return []

    flow_start_times = defaultdict(float)
    flow_end_times = defaultdict(float)
    flow_packets = defaultdict(list)
    flow_lengths = defaultdict(list)
    fwd_lengths = defaultdict(list)
    bwd_lengths = defaultdict(list)
    fwd_iats = defaultdict(list)
    bwd_iats = defaultdict(list)
    prev_time = None

    for pkt in cap:
        try:
            if hasattr(pkt, 'ip') and (hasattr(pkt, 'tcp') or hasattr(pkt, 'udp')):
                src_ip = pkt.ip.src
                dst_ip = pkt.ip.dst
                src_port = int(getattr(pkt, 'tcp', getattr(pkt, 'udp', pkt)).srcport) if (hasattr(pkt, 'tcp') or hasattr(pkt, 'udp')) else 0
                dst_port = int(getattr(pkt, 'tcp', getattr(pkt, 'udp', pkt)).dstport) if (hasattr(pkt, 'tcp') or hasattr(pkt, 'udp')) else 0
                flow_key = (src_ip, src_port, dst_ip, dst_port)

                curr_time = float(pkt.sniff_timestamp)

                # Cập nhật thời gian bắt đầu và kết thúc của luồng
                if flow_key not in flow_start_times:
                    flow_start_times[flow_key] = curr_time
                flow_end_times[flow_key] = curr_time

                # Tính IAT (Inter-Arrival Time)
                if prev_time is not None:
                    iat = curr_time - prev_time
                    flow_packets[flow_key].append(iat)

                # Phân loại gói tin (Fwd hoặc Bwd)
                is_fwd = (src_ip, src_port) == (pkt.ip.src, int(pkt[pkt.transport_layer].srcport))
                packet_length = int(pkt.length) if hasattr(pkt, 'length') else 0
                flow_lengths[flow_key].append(packet_length)

                if is_fwd:
                    fwd_lengths[flow_key].append(packet_length)
                    if len(fwd_lengths[flow_key]) > 1:
                        fwd_iats[flow_key].append(iat)
                else:
                    bwd_lengths[flow_key].append(packet_length)
                    if len(bwd_lengths[flow_key]) > 1:
                        bwd_iats[flow_key].append(iat)

                # Cập nhật luồng
                flow = flows[flow_key]
                flow['Total Fwd Packets'] += 1 if is_fwd else 0
                flow['Total Backward Packets'] += 1 if not is_fwd else 0
                flow['Total Length of Fwd Packets'] += packet_length if is_fwd else 0
                flow['Total Length of Bwd Packets'] += packet_length if not is_fwd else 0
                if is_fwd:
                    flow['Fwd Packet Length Max'] = max(flow['Fwd Packet Length Max'], packet_length)
                    flow['Fwd Packet Length Min'] = min(flow['Fwd Packet Length Min'], packet_length)
                else:
                    flow['Bwd Packet Length Max'] = max(flow['Bwd Packet Length Max'], packet_length)
                    flow['Bwd Packet Length Min'] = min(flow['Bwd Packet Length Min'], packet_length)

                # Cờ TCP
                if hasattr(pkt, 'tcp'):
                    flow['FIN Flag Count'] += 1 if pkt.tcp.flags_fin == '1' else 0
                    flow['SYN Flag Count'] += 1 if pkt.tcp.flags_syn == '1' else 0
                    flow['RST Flag Count'] += 1 if pkt.tcp.flags_reset == '1' else 0
                    flow['PSH Flag Count'] += 1 if pkt.tcp.flags_push == '1' else 0
                    flow['ACK Flag Count'] += 1 if pkt.tcp.flags_ack == '1' else 0

                prev_time = curr_time

        except AttributeError:
            continue

    # Tính toán các giá trị thống kê cho từng luồng
    result = []
    for flow_key, flow in flows.items():
        if flow['Total Fwd Packets'] + flow['Total Backward Packets'] > 0:
            # Tính Flow Duration
            start_time = flow_start_times[flow_key]
            end_time = flow_end_times[flow_key]
            duration = end_time - start_time if end_time > start_time else 1e-6  # Tránh duration = 0
            flow['Flow Duration'] = duration * 1e6  # Chuyển sang microgiây

            total_packets = flow['Total Fwd Packets'] + flow['Total Backward Packets']
            total_bytes = flow['Total Length of Fwd Packets'] + flow['Total Length of Bwd Packets']

            # Tính Flow Bytes/s và Flow Packets/s
            flow['Flow Bytes/s'] = total_bytes / duration if duration != 0 else 0
            flow['Flow Packets/s'] = total_packets / duration if duration != 0 else 0

            # Tính Packet Length Mean
            flow['Packet Length Mean'] = total_bytes / total_packets if total_packets > 0 else 0

            # Tính Packet Length Std
            lengths = flow_lengths[flow_key]
            if len(lengths) > 1:
                flow['Packet Length Std'] = np.std(lengths)
            else:
                flow['Packet Length Std'] = 0

            # Tính IAT
            iats = flow_packets[flow_key]
            if len(iats) > 0:
                flow['Flow IAT Mean'] = np.mean(iats) * 1e6
                flow['Flow IAT Std'] = np.std(iats) * 1e6
            else:
                flow['Flow IAT Mean'] = 0
                flow['Flow IAT Std'] = 0

            # Tính Fwd IAT Mean
            if len(fwd_iats[flow_key]) > 0:
                flow['Fwd IAT Mean'] = np.mean(fwd_iats[flow_key]) * 1e6
            else:
                flow['Fwd IAT Mean'] = 0

            # Tính Bwd IAT Mean
            if len(bwd_iats[flow_key]) > 0:
                flow['Bwd IAT Mean'] = np.mean(bwd_iats[flow_key]) * 1e6
            else:
                flow['Bwd IAT Mean'] = 0

            # Lưu thời gian bắt đầu của luồng (Timestamp) và định dạng thành YYYY-MM-DD HH:MM:SS
            flow['Timestamp'] = datetime.fromtimestamp(flow_start_times[flow_key]).strftime('%Y-%m-%d %H:%M:%S')

            # Xử lý dữ liệu: Giới hạn giá trị lớn hơn 99999999999
            for key in flow:
                if isinstance(flow[key], (int, float)) and key not in ['Timestamp', 'Label']:
                    flow[key] = 99999999999 if flow[key] > 99999999999 else flow[key]

            result.append(flow)

    return result

# Đường dẫn đến file .pcapng
pcap_file = r"D:\học tập\ai\attt\NHOM16_SOURCE\b_10_35.pcapng"  # Thay bằng đường dẫn thực tế của file

# Trích xuất các đặc trưng
features = extract_features(pcap_file)

# Chuyển thành DataFrame
df = pd.DataFrame(features, columns=network_features)

# Xử lý dữ liệu: Thay NaN bằng 0
df.fillna(0, inplace=True)

# Lưu thành file CSV
output_csv = r"D:\học tập\ai\attt\NHOM16_SOURCE\b.csv"
df = pd.read_csv(output_csv)
print(df['Label'].value_counts())
# print(f"Dataset has been saved to: {output_csv}")
# print(f"Dataset shape: {df.shape}")
# print(f"First few rows:\n{df.head()}")