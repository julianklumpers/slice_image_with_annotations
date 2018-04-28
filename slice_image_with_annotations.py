import cv2
import os
import xml.etree.ElementTree as ET
import math


def slice(image, xml, size=(5, 2), path='', suffix=''):

    height, width = image.shape[:2]
    wSize = int(math.ceil(float(height) / size[0]))

    for r in range(0, height, wSize):
        window = image[r:r+wSize]
        wHeight, wWidth = window.shape[:2]
        tSize = int(math.ceil(float(wWidth) / size[1]))
        for c in range(0, wWidth, tSize):
            tile = image[r:r + wSize, c:c + tSize]

            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'folder').text = 'images'
            ET.SubElement(annotation, 'filename').text = str(
                r) + '_' + str(c) + '_' + str(size[0]) + '_' + str(size[1]) + '_' + suffix + '.jpg'
            ET.SubElement(annotation, 'path').text = 'images/' + str(r) + '_' + \
                str(c) + '_' + str(size[0]) + '_' + \
                str(size[1]) + '_' + suffix + '.jpg'

            source = ET.SubElement(annotation, 'source')
            ET.SubElement(source, 'database').text = 'Unknown'

            imageSize = ET.SubElement(annotation, 'size')
            ET.SubElement(imageSize, 'width').text = str(tile.shape[0])
            ET.SubElement(imageSize, 'height').text = str(tile.shape[1])
            ET.SubElement(imageSize, 'depth').text = str(tile.shape[2])

            ET.SubElement(annotation, 'segmented').text = '0'

            y = r
            x = c
            xx = r + wSize
            yy = c + tSize

            for member in xml.findall('object'):
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)

                if(
                    xmin > x and xmax > x and
                    ymin > y and ymax > y and
                    xmin < yy and xmax < yy and
                    ymin < xx and ymax < xx
                ):
                    obj = ET.SubElement(annotation, 'object')
                    ET.SubElement(obj, 'name').text = member[0].text
                    ET.SubElement(obj, 'pose').text = member[1].text
                    ET.SubElement(obj, 'truncated').text = member[2].text
                    ET.SubElement(obj, 'difficult').text = member[3].text

                    bndbox = ET.SubElement(obj, 'bndbox')
                    ET.SubElement(bndbox, 'xmin').text = str(xmin - x)
                    ET.SubElement(bndbox, 'ymin').text = str(ymin - y)
                    ET.SubElement(bndbox, 'xmax').text = str(xmax - x)
                    ET.SubElement(bndbox, 'ymax').text = str(ymax - y)

            tree = ET.ElementTree(annotation)

            if tree.find('object') != None:

                tree.write(path + '/{}.xml'.format(str(r) + '_' + str(c) +
                                                   '_' + str(size[0]) + '_' + str(size[1]) + '_' + suffix))

                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

                cv2.imwrite(path + '/{}.jpg'.format(str(r) + '_' + str(c) + '_' + str(
                    size[0]) + '_' + str(size[1]) + '_' + suffix), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
