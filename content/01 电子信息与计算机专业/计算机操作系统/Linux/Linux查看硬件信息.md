在Linux服务器上查看硬件信息，可以通过以下命令获取CPU、内存、磁盘等重要参数：

### 1. 查看CPU信息
```bash
# 查看CPU型号
cat /proc/cpuinfo | grep "model name" | head -n 1

# 查看物理CPU个数
cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l

# 查看每个物理CPU的核心数
cat /proc/cpuinfo | grep "cpu cores" | head -n 1

# 查看总逻辑CPU数
nproc
```

### 2. 查看内存信息
```bash
# 查看内存总量
free -h

# 详细内存信息
dmidecode -t memory | grep -i size

# 查看内存插槽及使用情况
dmidecode -t memory | grep -A16 "Memory Device$"
```

### 3. 查看磁盘信息
```bash
# 查看磁盘分区
lsblk

# 查看磁盘使用情况
df -h

# 查看磁盘型号和详细信息
sudo smartctl -i /dev/sda
```

### 4. 查看主板和BIOS信息
```bash
# 主板信息
dmidecode -t baseboard

# BIOS信息
dmidecode -t bios
```

### 5. 查看网卡信息
```bash
# 网卡列表
lspci | grep -i ethernet

# 网卡详细信息
ip a
```

### 6. 查看GPU信息（如果有）
```bash
lspci | grep -i vga
nvidia-smi  # 如果是NVIDIA显卡
```

### 7. 综合信息查看工具
```bash
# 使用lshw综合查看（需要root权限）
sudo lshw -short

# 使用inxi工具（可能需要安装）
sudo apt install inxi  # Debian/Ubuntu
sudo yum install inxi  # CentOS/RHEL
inxi -Fxz
```

### 8. 查看服务器型号
```bash
# 查看服务器厂商和型号
dmidecode -t system | grep "Product Name"

# 查看序列号
dmidecode -t system | grep "Serial Number"
```

### 9. 查看RAID信息（如果有）
```bash
# MegaCli工具（需要安装）
sudo MegaCli -PDList -aAll
```

### 10. 查看温度传感器
```bash
sensors
```

### 一键获取主要硬件信息的脚本
```bash
#!/bin/bash

echo "===== 系统信息 ====="
cat /etc/os-release | grep PRETTY_NAME
uname -a

echo -e "\n===== CPU信息 ====="
echo "型号: $(cat /proc/cpuinfo | grep "model name" | head -n 1 | cut -d ":" -f 2 | sed 's/^[ \t]*//')"
echo "物理CPU数量: $(cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l)"
echo "每个CPU核心数: $(cat /proc/cpuinfo | grep "cpu cores" | head -n 1 | cut -d ":" -f 2 | sed 's/^[ \t]*//')"
echo "总逻辑CPU数: $(nproc)"

echo -e "\n===== 内存信息 ====="
free -h | grep "Mem:" | awk '{print "总内存: " $2, "已用: " $3, "可用: " $4}'

echo -e "\n===== 磁盘信息 ====="
lsblk | grep -v "loop"
df -h | grep -v tmpfs

echo -e "\n===== 网卡信息 ====="
ip a | grep -w "inet" | grep -v "127.0.0.1"

echo -e "\n===== 服务器型号 ====="
dmidecode -t system | grep "Product Name" | head -n 1
```

将上述脚本保存为`hardware_info.sh`，然后执行：
```bash
chmod +x hardware_info.sh
./hardware_info.sh
```

这些命令和脚本可以帮助您全面了解Linux服务器的硬件配置情况。根据您的具体需求，可以选择使用单个命令或组合使用多个命令来获取特定信息。