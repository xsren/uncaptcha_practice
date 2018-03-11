import os
import os.path

root_dir = '/home/stupid/hurong_captcha/GenPics'


for parent, dirnames, filenames in os.walk(root_dir):
    pics = []
    for filename in filenames:
        whole_name = parent + '/' + filename
        pics.append(whole_name)

# print (pics[1].split('/')[-1]).split('_')

print pics
# write to csv
def write_labels(pics):
    to_be_sorted = []
    for pic in pics:
        serial_num = (pic.split('/')[-1]).split('_')
        to_be_sorted.append(tuple(serial_num))

    result = sorted(to_be_sorted, key=lambda x: int(x[0]), reverse=False)
    labels = []
    for i in result:
        num = i[0]
        label = i[1][:4]
        labels.append([num, label])

    import csv
    csvfile = open('GenPics/labels.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(labels)
    csvfile.close()

def rename(pics):
    for pic in pics:
        if pic.endswith('.jpg'):
            new_filename = (pic.split('/')[-1]).split('_')[0]
            new_path = root_dir + '/' + new_filename + '.jpg'
            os.rename(pic, new_path)
        else:
            pass


write_labels(pics)
rename(pics)



