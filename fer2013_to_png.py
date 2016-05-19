import csv
from PIL import Image

#   settings
numbpics = 50
width = 48
height = 48
size = width,height
mode = 'RGB'
file = 'fer2013.csv'

#   read csv file
with open(file,'r') as csvin:
    dataset=csv.reader(csvin, delimiter=',', quotechar='"')
    rowcount=0
    for row in dataset:
        if rowcount > 0 and rowcount < numbpics+1:
            print 'picture ' + str(rowcount)
            x=0
            y=0
            pixels=row[1].split()
            img = Image.new(mode,size)
#   fill pixels left to right            
            for pixel in pixels:    
                colour=(int(pixel),int(pixel),int(pixel))
                img.putpixel((x,y), colour)
                x+=1
#   next pixelline after 48 pixels                
                if x >= width:         
                    x=0
                    y+=1
#   save picture as fer2013_picturenumber_emotionclass
            imgfile='fer2013_'+str(rowcount)+'_'+str(row[0])+'.png'
            img.save(imgfile,'png')
        rowcount+=1
