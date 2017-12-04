import xml.etree.ElementTree as ET
import glob, os, sys

target_counts = {}
if len(sys.argv) == 2:
	os.chdir(sys.argv[1])
else:
	print 'Usage: python target_classes.py training_data_directory'
	exit(1)

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


# identify targets with >100 occurrences
total_samples = 0
total_targets = 0
for key, value in sorted(target_counts.iteritems(), key=lambda(k,v): (v,k)):
	if not (key.isalpha() or key.isdigit()):
		print key + ': ' + str(value)
		total_samples += value
		total_targets += 1

print 'TOTAL SAMPLES: ' + str(total_samples)
print 'TOTAL TARGETS: ' + str(total_targets)
