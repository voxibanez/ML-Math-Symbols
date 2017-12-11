import os


cutoff = 1000;
counter = 0
counter2 = 0

root = "DataSet"
f = open(root + "/" + 'categories.txt','w')
for folder in os.listdir(root):
    if not os.path.isdir(root + "/" + folder):
        continue
    if folder.startswith('.'):
        continue
    folderFileCount = 0
    f.write(folder + "\n")
    for file in os.listdir(root + "/" + folder):
        folderFileCount = folderFileCount + 1;
    for file in os.listdir(root + "/" + folder):
        if file.startswith('.'):
            continue
        path, filename = os.path.split(file)
        if not os.path.exists("TRAIN_TRAIN/" + str(counter2)):
            os.makedirs("TRAIN_TRAIN/" + str(counter2))
        os.rename(root + "/" + folder + "/" + file, "TRAIN_TRAIN/" + str(counter2) + "/" + str(counter) + ".jpg")
        counter = counter + 1
        if(counter >= folderFileCount * 0.8):
            counter = 0
            break

    for file in os.listdir(root + "/" + folder):
        if file.startswith('.'):
            continue
        path, filename = os.path.split(file)
        if not os.path.exists("TEST_TEST/" + str(counter2)):
            os.makedirs("TEST_TEST/" + str(counter2))
        os.rename(root + "/" + folder + "/" + file, "TEST_TEST/" + str(counter2) + "/" + str(counter) + ".jpg")
        counter = counter + 1
        if(counter >= folderFileCount * (0.2)):
            counter = 0
            break
    counter2 = counter2 + 1
f.close()