import numpy as np 
from PIL import Image  
import random


class BucketManager:
    def __init__(self,minsize=256,maxsize=512,umbral=512*768,center=False):
        self.buckets,self.ratios=self.get_ResandRatio(minsize,maxsize,umbral) 
        self.center=center
        
    def get_ResandRatio(self,min=256,max=512,umbral=512*768):
        buckets=set()
        buckets
        width=min
        height=max
        while width<=max:
            height=max
            while height*width>umbral:
                height=height-64
            buckets.add((width,height))
            width=width+64
        height=min

        while height<=max:
            width=max
            while height*width>umbral:
                width=width-64
            buckets.add((width,height))
            height=height+64
            
        buckets.add((512,512))
        buckets=list(buckets)
        buckets.sort()
        bucketsratio=np.array(buckets)
        bucketsratio=bucketsratio[:,0]/bucketsratio[:,1]
        return buckets,bucketsratio

    def get_bucketid(self,img):
        ratio=img.size[0]/img.size[1]
        id=np.argmin(abs(self.ratios-ratio))
        error=abs(self.ratios[id]-ratio)
        if error<4:
            return id
        return -254
    
    def get_bucketdata(self,id):
        return self.buckets[id],self.ratios[id]
    
    def process_image(self,img,id=-1):
        
        if id==-1:
            id=self.get_bucketid(img)
        ix,iy=img.size
        imgratio=ix/iy
        bx,by=self.buckets[id]
        ratio=self.ratios[id]
         
        if imgratio==ratio:
            newx=bx
            newy=by
            img=img.resize((newx,newy),Image.LANCZOS)
            #print("resize")
            return img
        
        elif imgratio>ratio:
            newx=int(by*imgratio)
            newy=by
            
        else:
            newx=bx
            newy=int(bx/imgratio)
            
        img=img.resize((newx,newy),Image.LANCZOS)
        if self.center:
            img=self.__center_crop(img,(bx,by))
        else:
            img=self.__random_crop(img,(bx,by)) 
        return img
    
    def __random_crop(self,img, crop_size):
        width, height = img.size
        crop_width, crop_height = crop_size
        if width < crop_width or height < crop_height:
            raise ValueError("Crop size should be less than image size")
        random_width = random.randint(0, width - crop_width)
        random_height = random.randint(0, height - crop_height)
        cropped_img = img.crop((random_width, random_height, random_width + crop_width, random_height + crop_height))
        return cropped_img 
    
    def __center_crop(self,img, crop_size):
        width, height = img.size
        crop_width, crop_height = crop_size

        if width < crop_width or height < crop_height:
            raise ValueError("Crop size should be less than image size")

        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        right = (width + crop_width) / 2
        bottom = (height + crop_height) / 2

        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img


from torch.utils.data import Sampler    

class BucketBatchSampler(Sampler):
    def __init__(self, bucket_ids, batch_size, shuffle=False):
        self.bucket_ids = bucket_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        # Agrupar índices por bucket
        buckets = {}
        for idx, bucket_id in enumerate(self.bucket_ids):
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(idx)

        # Devolver índices en lotes, bucket por bucket
        for bucket_id in buckets:
            indices = buckets[bucket_id]
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i+self.batch_size]

    def __len__(self):
        return len(self.bucket_ids) // self.batch_size
