# Lập trình song song trên GPU - Đồ án cuối kỳ
* Thành viên:  
    * Ngô Bá Hoàng Thiên - 1612649
    * Nguyễn Quang Phú  - 1612508
* Quá trình thực hiện:
    * 9/12/2019: Thảo luận về đồ án, lên kế hoạch làm việc, an ủi, động viên nhau.
    * 10/12/2019-20/12/2019 : Thực hiện cài radix-sort tuần tự.
    * 03/01/2020-10/01/2019 : Thực hiện cài radix-sort song song.
* Các phiên bản:
    * Baseline_1.cu: Cài đặt tuần tự thuật toán Radix Sort tuần tự
    * Baseline_2.cu: Cài đặt song song 2 bước histogram và scan của thuật toán Radix Sort tuần tự
    * Baseline_3_v1.cu: Cài đặt tuần tự thuật toán Radix Sort với k = 1 bit
    * Baseline_3_v2.cu: Cài đặt song song thuật toán Radix Sort với k = 1 bit (song song phần exclusive scan được mảng inScan)
    * Baseline_3_v3.cu: Cài đặt song song thuật toán Radix Sort với k = 1 bit
    * Baseline_4_v1.cu: Cài đặt tuần tự thuật toán Radix Sort
    * Baseline_4_v2.cu: Cài đặt song song thuật toán Radix Sort (song song hóa phần tính localHost)
    * Baseline_4_v3.cu: Cài đặt song song thuật toán Radix Sort (song song hóa phần tính localHost,exclusive scan)
    * Baseline_4_v4.cu: Cài đặt song song thuật toán Radix Sort (song song hóa phần tính localHost,exclusive scan, tính radix sort 1 bit)
    * Baseline_4_v5.cu: Cài đặt song song thuật toán Radix Sort (song song hóa phần tính localHost,exclusive scan, tính radix sort của n bit)
    * Baseline_4_v6.cu: Cài đặt song song thuật toán Radix Sort (song song hóa phần tính localHost,exclusive scan, tính radix sort của n bit, tính start)
    * Baseline_4_v7.cu: Cài đặt song song thuật toán Radix Sort (song song hóa phần tính localHost,exclusive scan, tính radix sort của n bit, tính start, tính scatter)
    * Baseline_4_v8.cu: Bỏ các phần memcpy thừa từ host vào device và device sang host
    * Baseline_4_v9.cu: Chỉnh lại exclusive scan
