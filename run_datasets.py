import os, glob, sys
from parseG import parse
from parseKRON import parse
import pickle

if __name__ == "__main__":
    # dir_name = sys.argv[1]
    # for dirc in os.listdir(dir_name):
    #     if dirc[0] == '.':
    #         continue

    #     path = os.path.join(dir_name, dirc, '')
    #     if len(glob.glob(path + "*.edge_list")) == 0:
    #         if dirc[0] == 'kron':
    #             print('found a kronnecker graph parsing..............')
    #             parse(path, dirc)
    #         else:
    #             print("parsing.....................", dirc)
    #             parse(path, dirc)

    #     print("testing......................", dirc)
    #     # os.system("python3 Hierarchy_Topology_MAIN.py %s"%dirc)
        # os.system("python3 Hierarchy_GAN_TOPOLOGY_MAIN.py --Dataset %s --input_partition_dir %s --training_info_dir %s_partition_info.pickle"%(dirc, dirc, dirc))

    """Now We Have: ["BA","kron","WS","ER","wiki-vote","facebook"]"""
    # dirc = sys.argv[1] # WS has DONE!
    for dirc in ["BA","kron","WS","ER","facebook","wiki-vote"]:
    # for dirc in ["wiki-vote"]:
        print("#############################################")
        print("\ntesting......................", dirc)
        print("\n#############################################")
        if os.path.exists('%s_cut_value_list.pickle'%dirc):
            flatten_value = pickle.load(open('%s_cut_value_list.pickle'%dirc,'rb'))
        else:
            flatten_value = [10]
        for cutvalue in flatten_value:
            os.system("python3 Hierarchy_GAN_TOPOLOGY_MAIN.py --Dataset %s --input_partition_dir %s --training_info_dir %s_partition_info.pickle --cutValue %f"%(dirc, dirc, dirc, cutvalue))

