# Hierarchical Graph Generator
Using GAN to generate arbitrary topology

## Usage
### STEP.0 Transform Graph as edge_list Format
Please pay attension that the Graph should be in CONTINUOUS NUMBER
```
>>> Creat Graph in folder "data": data/<filename>.edge_list
    --- <filename>.edge_list Format: src dst
```

### STEP.1 Prepared Needed Information for Hierarchical Graph Generator
```
>>> python3 Hierarchy_Topology_MAIN.py <filename>.edge_list
```

### STEP.2 Running Hierarchical Graph Generator
```
>>> python3 Hierarchy_GAN_TOPOLOGY_MAIN.py  --Dataset <filename> --training_info_dir <filename>_partition_info.pickle --input_partition_dir <filename>
```

### Results:
```
>>> Reconstructed Graph located in reconstruction/ filname / Hierarchy/ reconstructed_<filename>.adj
```

## p.s. Metis Tool Introduction

```
./gpmetis <filename> <partition_num>
    or
./gpmetis -config <filename> <partition_num>
```
