
# bs
# ./bs --device 3 --dataset ../../dataset/skew_ratio/10000_10000_1000_1900000.s_data

# ip
# ./ip --device 3 --dataset ../../dataset/skew_ratio/10000_10000_1000_1900000.s_data

# hi
# ./hi --device 3 --buckets 12000 --bucket_size 100 --dataset ../../dataset/skew_ratio/10000_10000_1000_1900000.s_data

# bi_naive
# ./bi_naive --device 3 --dataset ../../dataset/skew_ratio/10000_10000_1000_1900000.s_data

# bi_dynamic
# ./bi_dynamic --device 3 --dataset ../../dataset/skew_ratio/10000_10000_1000_1900000.s_data 
# ./bi_dynamic --device 3 --dataset ../../dataset/skew_ratio/10000_5000000_1000_500900000.s_data --blocks 200

# rt
# ./rt_intersection --device 3 --chunk_length 3000 --dataset ../../dataset/10000_10000_1000_1900000.s_data