import xml.etree.ElementTree as ET
import glob, os, sys

def gen_target_classes(training_dir, thresh):
    target_counts = {}
    os.chdir(training_dir)

    for file in glob.glob('*.inkml'):
        tree = ET.parse(file)
        root = tree.getroot()
        trace_group = root[-1]

        targets = []
        for child in trace_group[1:]:
            current = child[0].text
            targets.append(current)
            if current not in target_counts:
                target_counts[current] = 1
            else:
                target_counts[current] += 1

    # identify targets with >thresh occurrences
    total_samples = 0
    total_targets = 0
    target_list = []
    for key, value in sorted(target_counts.iteritems(), key=lambda(k,v): (v,k)):
        if value > thresh and not (key.isalpha() or key.isdigit()):
            target_list.append(key)
            total_samples += value
            total_targets += 1

    return target_list, total_samples, total_targets

#if len(sys.argv) == 3:
 #   target_list, samples, targets = gen_target_classes(sys.argv[1], int(sys.argv[2]))
#else:
 #   print('Usage: python target_classes.py training_data_directory threshold')

#for t in target_list:
 #   print t

#print 'TOTAL SAMPLES: ' + str(samples)
#print 'TOTAL TARGETS: ' + str(targets)
