import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
from copy import deepcopy
from sortedcontainers import SortedDict

k=10				#number of children
p = 6				#no of children for active search
Max_number = 500		#max no of values in a leaf node
word_radius  = 100		#represents size of words in dictionary
cluster_no = 16			#no of words formed with new image
prob_lc = np.array(1)
words = {}
common_words = {}
count_cw=0
count_cw_wordid=0
word_id=0


def setcommonwords(wordid,ct):
	global common_words
	global count_cw
	global count_cw_wordid
	if(len(common_words)<cluster_no or wordid in common_words):
		common_words[wordid] = words[wordid]
		least= float("inf")
		leastid = 0
		for key in common_words:
			val = common_words[key]
			if (val[1]<least):
				least = val[1]
				leastid = key
		count_cw = least
		count_cw_wordid = leastid

		
		
	elif(ct>count_cw):
		common_words[wordid] = words[wordid]
		del common_words[count_cw_wordid]
		least= float("inf")
		leastid = 0
		for key in common_words:
			val = common_words[key]
			if (val[1]<least):
				least = val[1]
				leastid = key
		count_cw = least
		count_cw_wordid = leastid
				
		
	

def setwords(word,ct,wordid):
	global word_id
	global words
	if wordid in words:
		words[wordid][1] = ct
	else:
		words[wordid] = [word,ct]
		word_id+=1
	setcommonwords(wordid,ct)



def find_word_radius(kmeans,des):
	dist=np.zeros(len(kmeans.cluster_centers_))
	i=0
	for ele in kmeans.labels_:
		d=np.linalg.norm(kmeans.cluster_centers_[ele]- des[i])
		i+=1
		if dist[ele]<d:
			dist[ele]=d
		
	return dist
	
def find_word_radius_split(kmeans,des,val_imno,node):
	global common_words
	dist=np.zeros(len(kmeans.cluster_centers_))
	img_list=[]
	for no in range(len(kmeans.cluster_centers_)):
		img_list.append([])
	i=0
	for ele in kmeans.labels_:
		if(node.val_wordid[i] in common_words):
			del common_words[node.val_wordid[i]]
		d=np.linalg.norm(kmeans.cluster_centers_[ele]- des[i])
		img_list[ele]+= val_imno[i]
		
		i+=1
		if dist[ele]<d:
			dist[ele]=d
		
		
	return dist,img_list

	

def findL2Norm(f,arr):
	dmatch_list=[]
	no=0
	for item in arr:
		bf = cv2.BFMatcher()
		matches=bf.match(f,item.reshape(1,128))
		matches[0].trainIdx = no
		dmatch_list.append(matches[0])
		no+=1
	return dmatch_list
	
class Node:
	def __init__(self,keyval,parent,imno):
		self.key = keyval.reshape(1,128)
		self.wordid = word_id
		self.parent = parent
		self.children = []
		self.values = []
		self.childrenvals = []
		self.val_sum=0
		self.imno=imno
		self.val_imno=[]
		self.val_wordid=[]
		setwords(self.key,len(self.imno),word_id)
		
	def insert(self,value,rad,imno):
		global word_radius
		global no_words
		#search
		self.val_sum+=rad
		word_radius = (word_radius*no_words + rad)/(no_words+1)
		no_words+=1
		if len(self.values)+1 < Max_number:
			
			
			if len(self.values)==0:
				self.values = value.reshape(1,128)
			else:
				self.values = np.append(self.values,value.reshape(1,128),axis=0)
			#print (len(self.values) ,Max_number)
			self.val_imno.append([imno])
			self.val_wordid.append(word_id)
			setwords(value.reshape(1,128),1,word_id)
		else:
			#kmeans 
			self.values = np.append(self.values,value.reshape(1,128),axis=0)
			self.val_wordid.append(word_id)
			self.val_imno.append([imno])
			setwords(value.reshape(1,128),1,word_id)

			#print ("split leaf node",len(self.values) ,Max_number)
			
			kmeans = KMeans(n_clusters=10).fit(self.values)
			dist,img_list=find_word_radius_split(kmeans,self.values,self.val_imno,self)
			word_radius = (word_radius*no_words - self.val_sum + sum(dist))/(no_words-len(self.values)+len(dist))
			no_words = no_words-len(self.values)+len(dist)
			self.val_sum=0
			ct=0
			for c_center in kmeans.cluster_centers_:
				#print ("here")
				temp = Node(c_center,self,img_list[ct])
				ct+=1
				self.children.append(temp)
				if len(self.childrenvals) == 0:
					self.childrenvals = c_center.reshape(1,128)
				else :
					self.childrenvals = np.append(self.childrenvals,c_center.reshape(1,128),axis=0)
			#print("len of children",len(self.childrenvals))
			self.values = None
			self.val_imno=0
			
	def search_insert(self,f,imageno):
		global word_radius
		word_list = None
		min_dist = math.inf
		# compare if 0 return
		bf = cv2.BFMatcher()
		matches = bf.match(f.reshape(1,128),self.key)
				
		if(matches[0].distance < word_radius):
			word_list = None
			self.imno.append(imageno)
			setwords(self.key,len(self.imno),self.wordid)
			return word_list
		elif (self.values != None): # self.children is empty then leaf
			#print('at 6')
			if(len(self.values)!=0):
				bf = cv2.BFMatcher()
				#print (self.values.shape)
				matches = bf.match(f.reshape(1,128),self.values)
				#print ("-----------------------------",len(matches),self.values.shape)
				# dont think i need to sort now 
				matches = sorted(matches, key = lambda x:x.distance)
				if(matches[0].distance < word_radius):
					self.val_imno[matches[0].trainIdx].append(imageno)
					#print("at 1")
					setwords(self.values[matches[0].trainIdx],len(self.val_imno[matches[0].trainIdx]),self.val_wordid[matches[0].trainIdx])
					return None
				else:
					#print ('at 4')
					return self
			else:
				return self
		else:
			bf = cv2.BFMatcher()
			matches = findL2Norm(f.reshape(1,128),self.childrenvals)
			#print ("children::::::::::::",len(matches))
			matches = sorted(matches, key = lambda x:x.distance)
			ext_word = 0
			for match in matches[:p]:
				
				#print(match.distance,match.trainIdx)
				if (match.distance < word_radius):
					self.children[match.trainIdx].imno.append(imageno)
					setwords(self.children[match.trainIdx].key,len(self.children[match.trainIdx].imno),self.children[match.trainIdx].wordid)
					return None
				else:
					ext_word = 1
					temp = self.children[match.trainIdx].search_insert(f,imageno)
					if (temp == None):
						#print("at 2")
						return temp
					elif min_dist >  match.distance:
						word_list = temp
				

		return word_list
		
		
	def search(self,f):
		word_list = []
		img_list = []
		# compare if 0 return
		bf = cv2.BFMatcher()
		matches = bf.match(f.reshape(1,128),self.key)
		
		if(matches[0].distance < word_radius):
			word_list.append(self.key)
			img_list+=self.imno
		if (self.values != None): # self.children is empty then leaf
			#print('at 6')
			if(len(self.values)!=0):
				#bf = cv2.BFMatcher()
				#print (self.values.shape)
				matches = findL2Norm(f.reshape(1,128),self.values)
				#print ("-----------------------------",len(matches),self.values.shape)
				#matches = sorted(matches, key = lambda x:x.distance)
				ct=0
				for match in matches:
					if(match.distance < word_radius):
						#print("at 1")
						word_list.append(self.values[match.trainIdx])
						img_list += self.val_imno[ct]
						#return None
					ct+=1
			
		else:
			bf = cv2.BFMatcher()
			matches = findL2Norm(f.reshape(1,128),self.childrenvals)
			#print ("children::::::::::::",len(matches))
			matches = sorted(matches, key = lambda x:x.distance)
			for match in matches[:p]:
				
				#print(match.distance,match.trainIdx)
				if (match.distance < word_radius):
					temp,imlist = self.children[match.trainIdx].search(f)
					word_list += temp
					img_list+=imlist
					


		return word_list,img_list
		
	
	def getKey(self):
		return self.key
	def getParent(self):
		return self.parent
	def getChildren(self):
		return self.children
	def getChild(self,i):
		return self.children[i-1]
	def getValues(self):
		return self.values
	def getValue(self,i):
		return self.values[i-1]
	def AddValues(self,value,imno):
		if (len(self.values)==0):
			self.values = value.reshape(1,128)
			
		else:
			self.values  = np.append(self.values,value.reshape(1,128),axis=0)
		self.val_imno.append([imno])
		

def findnlc(c_center):
	rlist=[]
	for key in common_words:
		val=common_words[key]
		if np.linalg.norm(val[0]-c_center) < word_radius:
			rlist.append(-1)
	return rlist
	
def likelihood(kmeans,Dictionary,imno):
	score_list = np.zeros(imno+1)
	for c_center in kmeans.cluster_centers_:
		leaf_insertion,il = Dictionary.search(c_center)
		il = il + findnlc(c_center)
		if len(il)!=0:
			tf = (1.0/len(kmeans.cluster_centers_))*math.log(float(imno)/len(set(il)))
		for ele in il:
			if(ele<imno):
				score_list[ele+1] += tf
	if sum(score_list)==0:
		
		return np.ones(imno+1)
	mean_val = np.mean(score_list)
	std = np.std(score_list)
	cnt=0
	for score in score_list:
		if score>= mean_val+std:
			score_list[cnt]=(score-std)/float(mean_val)
		else:
			score_list[cnt]=1
		cnt+=1
	return score_list
			
		
def findtev(i,j):
	if i==-1 and j == -1:
		return .9
	elif i!=-1 and j==-1:
		return .1/imno
	elif i==-1 and j!=-1:
		return .1
	elif i-j==0:
		return 0.39894347
	elif abs(i-j)==1:
		return 0.24210528
	elif abs(i-j)==2:
		return 0.05842299
	else:
		return 0
	
		

cap = cv2.VideoCapture('vid.mp4')
no=0
# Flag for dictionary creation
dict_creation = 0
no_words = 0
Dictionary = None
imno=0
while(True):
	
	#try:
	ret, frame = cap.read()
	# Check if video exists
	if ret == False:
		break
	if(no%15==0):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp,des  = sift.detectAndCompute(gray,None)
		img=cv2.drawKeypoints(gray,kp,frame)
		
		
		# cluster keypoints 
		if des == None:
			continue
		if  len(des)<cluster_no:
			continue
		#print("sift features:::",des.shape)
		kmeans = KMeans(n_clusters=cluster_no).fit(des)

		if Dictionary == None:
			prob_lc = np.append(prob_lc,0)
			dict_creation = 1
			Dictionary = Node(kmeans.cluster_centers_[0],None,[imno])
			
			dist = find_word_radius(kmeans,des)
			'''
			word_radius = sum(dist)/len(dist)
			no_words= len(dist)
			'''
			word_radius = dist[0]
			no_words = 1
			d_val=1
			
			for c_center in kmeans.cluster_centers_[1:]:
				leaf_insertion = Dictionary.search_insert(c_center,imno)
				if (leaf_insertion != None):
					leaf_insertion.insert(c_center,dist[d_val],imno)
				d_val+=1
			
		else:
			dist = find_word_radius(kmeans,des)
			d_val = 0
			# likelihood calculation
			if (imno>10):
				likelihood_score = likelihood(kmeans,Dictionary,imno-10)
				prob_lc_n = np.zeros(prob_lc.shape[0])
				for it in range(imno+1-10):
					ptemp=0
					for ite in range(imno+1-10):
						lval = findtev(it-1,ite-1)
						ptemp+= lval*prob_lc[ite]
					prob_lc_n[it] = likelihood_score[it]*ptemp
				prob_lc_n = prob_lc_n/(sum(prob_lc_n)+0.0)
				so=0
				so_id=0
				
				for i in range(len(prob_lc_n)-5):
					temp = sum(prob_lc_n[i:i+5])
					if (so<temp):
						so=temp
						so_id = i
				
				#if (np.argmax(prob_lc_n)!=0):
				print ("loop closure with image: ",so_id,"prob:",so,"at image",imno )
				prob_lc = deepcopy(prob_lc_n)
			
			prob_lc = np.append(prob_lc,0)
			for c_center in kmeans.cluster_centers_:
				leaf_insertion = Dictionary.search_insert(c_center,imno)
				if (leaf_insertion != None):
				
					leaf_insertion.insert(c_center,dist[d_val],imno)
				d_val+=1
		
		imno+=1
		cv2.imwrite("data/im"+str(imno)+".jpg",img)
		cv2.imshow('frame',img)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
	no+=1

	
	
cap.release()
cv2.destroyAllWindows()
print (no)
		
		
